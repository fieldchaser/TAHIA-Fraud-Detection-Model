import json
import os
import time
import urllib.error
import urllib.request

import numpy as np
import torch
from sklearn.metrics import accuracy_score, average_precision_score, confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score


def _to_float(x):
    if isinstance(x, dict):
        return {str(k): _to_float(v) for k, v in x.items()}
    if isinstance(x, range):
        return {"start": x.start, "stop": x.stop, "step": x.step}
    if isinstance(x, np.ndarray):
        return _to_float(x.tolist())
    if isinstance(x, (list, tuple)):
        return [_to_float(v) for v in x]
    if isinstance(x, (np.floating, np.integer)):
        return float(x)
    return x


def _eval_split(logits, index, labels):
    probs = torch.softmax(logits[index], dim=1).detach().cpu().numpy()[:, 1]
    pred = torch.argmax(logits[index], dim=1).detach().cpu().numpy()
    target = labels.detach().cpu().numpy()
    return {
        "auc": float(roc_auc_score(target, probs)),
        "auprc": float(average_precision_score(target, probs)),
        "precision_macro": float(precision_score(target, pred, average="macro", zero_division=0)),
        "recall_macro": float(recall_score(target, pred, average="macro", zero_division=0)),
        "f1_macro": float(f1_score(target, pred, average="macro", zero_division=0)),
        "accuracy": float(accuracy_score(target, pred)),
        "confusion_matrix": confusion_matrix(target, pred).tolist(),
    }


def _curve_summary(values, tail=10):
    if not values:
        return {}
    vals = [float(v) for v in values]
    head_val = vals[0]
    tail_vals = vals[-tail:]
    best_val = min(vals)
    return {
        "first": head_val,
        "last": vals[-1],
        "best": best_val,
        "delta_first_last": vals[-1] - head_val,
        "tail_mean": float(np.mean(tail_vals)),
        "tail_std": float(np.std(tail_vals)),
        "length": len(vals),
    }


def _metric_summary(metric_curve, key, tail=10):
    if not metric_curve:
        return {}
    vals = [float(x[key]) for x in metric_curve if key in x]
    if not vals:
        return {}
    tail_vals = vals[-tail:]
    return {
        "first": vals[0],
        "last": vals[-1],
        "best": max(vals),
        "delta_first_last": vals[-1] - vals[0],
        "tail_mean": float(np.mean(tail_vals)),
        "tail_std": float(np.std(tail_vals)),
        "length": len(vals),
    }


def _summarize_training_trace(training_trace):
    if not training_trace:
        return {}
    val_metrics = training_trace.get("val_metrics", [])
    return {
        "train_loss": _curve_summary(training_trace.get("train_loss", [])),
        "val_loss": _curve_summary(training_trace.get("val_loss", [])),
        "val_auc": _metric_summary(val_metrics, "auc"),
        "val_precision_macro": _metric_summary(val_metrics, "precision_macro"),
        "val_recall_macro": _metric_summary(val_metrics, "recall_macro"),
        "val_f1_macro": _metric_summary(val_metrics, "f1_macro"),
        "last_val_metrics": val_metrics[-5:],
    }


def _label_profile(split_name, labels):
    y = labels.detach().cpu().numpy()
    keys, counts = np.unique(y, return_counts=True)
    total = int(len(y))
    return {
        "split": split_name,
        "total": total,
        "counts": {str(int(k)): int(v) for k, v in zip(keys, counts)},
        "ratio": {str(int(k)): float(v / total) for k, v in zip(keys, counts)},
    }


def _diagnose(payload):
    metrics = payload["metrics"]
    trace = payload.get("training_summary", {})
    test = metrics["test"]
    val = metrics["val"]
    diagnoses = []

    train_loss = trace.get("train_loss", {})
    val_loss = trace.get("val_loss", {})
    if train_loss and val_loss:
        if train_loss.get("delta_first_last", 0.0) >= -1e-3:
            diagnoses.append("loss_not_decreasing")
        elif val_loss.get("last", 0.0) > val_loss.get("best", 0.0) * 1.05:
            diagnoses.append("possible_overfitting")

    if test["recall_macro"] + 0.05 < test["precision_macro"]:
        diagnoses.append("recall_low")
    if test["precision_macro"] + 0.05 < test["recall_macro"]:
        diagnoses.append("precision_low")
    if test["auprc"] < test["auc"] - 0.1:
        diagnoses.append("class_imbalance_or_ranking_issue")
    if val["f1_macro"] > test["f1_macro"] + 0.08:
        diagnoses.append("validation_test_gap")
    if not diagnoses:
        diagnoses.append("stable_or_near_optimal")
    return diagnoses


def _collect_dap_gates(model):
    gates = {}
    for name, module in model.named_modules():
        if hasattr(module, "dap_gate"):
            gates[name] = float(torch.sigmoid(module.dap_gate.detach()).cpu())
    return gates


def _rule_feedback(payload):
    val = payload["metrics"]["val"]
    test = payload["metrics"]["test"]
    params = payload["params"]
    use_dap = params.get("use_dap_encoder", True)
    gate_init = float(params.get("dap_gate_init", -3.0))
    loss = payload["loss_weight"]

    diagnoses = payload.get("diagnosis_tags", [])
    suggestions = []
    if "precision_low" in diagnoses or test["precision_macro"] < test["recall_macro"]:
        suggestions.append({
            "change": "increase precision",
            "next_run": {
                "dap_gate_init": gate_init - 0.5 if use_dap else gate_init,
                "loss_w0": round(min(loss[0] + 0.02, 0.45), 4),
                "loss_w1": round(max(loss[1] - 0.02, 0.55), 4),
            },
            "reason": "macro precision is lower than macro recall, so the model may be over-predicting the fraud class.",
        })
    elif "recall_low" in diagnoses or test["recall_macro"] < test["precision_macro"]:
        suggestions.append({
            "change": "recover recall",
            "next_run": {
                "dap_gate_init": gate_init + 0.5 if use_dap else gate_init,
                "loss_w0": round(max(loss[0] - 0.02, 0.02), 4),
                "loss_w1": round(min(loss[1] + 0.02, 0.98), 4),
            },
            "reason": "macro recall is lower than macro precision, so the model may be too conservative.",
        })

    if use_dap:
        suggestions.append({
            "change": "DAP gate local search",
            "next_run": {
                "dap_gate_init": gate_init,
                "dap_bins": params.get("dap_bins", 8),
                "dap_temp": params.get("dap_temp", 0.2),
            },
            "reason": "keep the current gated residual DAP as the anchor and only search nearby gate values.",
        })
    else:
        suggestions.append({
            "change": "try gated residual DAP",
            "next_run": {
                "dap_gate_init": -4.0,
                "dap_bins": 8,
                "dap_temp": 0.2,
                "dap_dropout": 0.0,
            },
            "reason": "the diagnostic module can test whether dynamic attribute patterns improve F1 on this dataset.",
        })

    return {
        "diagnosis": (
            "Rule-based fallback feedback. No external LLM API key was used, so this is generated locally from metrics, "
            "confusion matrices, loss weights, attention weights, and DAP gate values."
        ),
        "metric_focus": "macro F1",
        "diagnosis_tags": diagnoses,
        "next_run_suggestions": suggestions[:3],
        "paper_description": (
            "The LLM diagnostic feedback module is attached after model evaluation. It analyzes validation/test "
            "metrics, error distribution, fusion weights, and DAP gates, then recommends non-invasive tuning signals "
            "for the feature mapping and loss configuration."
        ),
    }


def _build_prompt(payload):
    return (
        "You are an LLM-guided training diagnosis module for a graph fraud detection model. "
        "First classify the training issue, then suggest bounded auto-tuning actions. "
        "Do not change the model architecture. Focus on macro F1 and AUPRC under class imbalance. "
        "Analyze the following run summary and return valid JSON only. "
        "Do not wrap the JSON in markdown fences. "
        "The JSON must contain keys: diagnosis, issue_type, evidence, next_run_suggestions, and paper_description. "
        "Choose issue_type from: overfitting, recall_low, precision_low, loss_not_decreasing, "
        "class_imbalance_or_auprc_low, stable_or_near_optimal. "
        "Only dap_gate_init, loss_w0, and loss_w1 may be suggested for the next run. "
        "Do not suggest dap_bins, dap_temp, or dap_dropout because they are fixed. "
        "Each next_run_suggestion must contain concrete parameter values and a short rationale. "
        "The maximum allowed absolute changes are: dap_gate_init <= 0.5, loss_w0 <= 0.02, loss_w1 <= 0.02. "
        "Use this exact shape for suggestions: "
        '{"next_run_suggestions":[{"change":"...","next_run":{"dap_gate_init":-4.0,"loss_w0":0.06,"loss_w1":0.94},"reason":"..."}]}'
        "\n\n"
        f"{json.dumps(payload, ensure_ascii=False, indent=2)}"
    )


def _parse_json_text(text):
    if not text:
        return None
    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = stripped.strip("`")
        if stripped.startswith("json"):
            stripped = stripped[4:].strip()
    try:
        return json.loads(stripped)
    except json.JSONDecodeError:
        pass

    start = stripped.find("{")
    end = stripped.rfind("}")
    if start >= 0 and end > start:
        try:
            return json.loads(stripped[start:end + 1])
        except json.JSONDecodeError:
            return None
    return None


def _as_float(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _complete_binary_loss_weights(cleaned):
    if "loss_w0" in cleaned and "loss_w1" not in cleaned:
        value = _as_float(cleaned.get("loss_w0"))
        if value is not None:
            cleaned["loss_w1"] = round(1.0 - value, 6)
    elif "loss_w1" in cleaned and "loss_w0" not in cleaned:
        value = _as_float(cleaned.get("loss_w1"))
        if value is not None:
            cleaned["loss_w0"] = round(1.0 - value, 6)
    return cleaned


def extract_next_run(feedback, base_params=None, max_loss_delta=0.02, max_gate_delta=0.5):
    if not feedback:
        return {}

    parsed = feedback.get("parsed") if isinstance(feedback, dict) else None
    if parsed is None and isinstance(feedback, dict) and "text" in feedback:
        parsed = _parse_json_text(feedback.get("text", ""))
    source = parsed if isinstance(parsed, dict) else feedback

    suggestions = source.get("next_run_suggestions", []) if isinstance(source, dict) else []
    allowed = {
        "dap_gate_init",
        "loss_w0",
        "loss_w1",
    }
    for item in suggestions:
        if not isinstance(item, dict):
            continue
        next_run = item.get("next_run", {})
        if not isinstance(next_run, dict):
            continue
        cleaned = {k: v for k, v in next_run.items() if k in allowed}
        cleaned = _complete_binary_loss_weights(cleaned)
        if base_params is not None:
            eps = 1e-8
            loss_invalid = False
            if "loss_w0" in cleaned:
                base = _as_float(base_params.get("loss_w0"))
                value = _as_float(cleaned.get("loss_w0"))
                if base is None or value is None or abs(value - base) > max_loss_delta + eps:
                    loss_invalid = True
            if "loss_w1" in cleaned:
                base = _as_float(base_params.get("loss_w1"))
                value = _as_float(cleaned.get("loss_w1"))
                if base is None or value is None or abs(value - base) > max_loss_delta + eps:
                    loss_invalid = True
            if loss_invalid:
                cleaned.pop("loss_w0", None)
                cleaned.pop("loss_w1", None)
            if "dap_gate_init" in cleaned:
                base = _as_float(base_params.get("dap_gate_init"))
                value = _as_float(cleaned.get("dap_gate_init"))
                if base is None or value is None or abs(value - base) > max_gate_delta + eps:
                    cleaned.pop("dap_gate_init", None)
        if cleaned:
            return cleaned
    return {}


def _provider_defaults(provider):
    if provider == "qwen":
        return {
            "model": "qwen-plus",
            "base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1",
            "api_key_env": "DASHSCOPE_API_KEY",
        }
    return {
        "model": "gpt-5.4-mini",
        "base_url": "https://api.openai.com/v1",
        "api_key_env": "OPENAI_API_KEY",
    }


def _call_llm(prompt, provider, model, base_url, api_key_env):
    api_key = os.environ.get(api_key_env)
    if not api_key:
        return None


    body = {
        "model": model,
        "messages": [
            {
                "role": "system",
                "content": "You produce concise diagnostic feedback for graph fraud detection experiments.",
            },
            {
                "role": "user",
                "content": prompt,
            },
        ],
    }
    request = urllib.request.Request(
        base_url.rstrip("/") + "/chat/completions",
        data=json.dumps(body).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=60) as response:
            data = json.loads(response.read().decode("utf-8"))
    except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError) as exc:
        return {"error": str(exc)}

    choices = data.get("choices", [])
    text = ""
    if choices:
        text = choices[0].get("message", {}).get("content", "")
    parsed = _parse_json_text(text)
    return {
        "provider": provider,
        "model": model,
        "base_url": base_url,
        "text": text,
        "parsed": parsed,
        "raw_id": data.get("id"),
    }


def run_llm_feedback(
    cf, model, logits, test_x, test_y, val_x, val_y, stopper, best_weights,
    loss_weight, args, data_profile=None, training_trace=None, runtime_profile=None
):
    if not getattr(args, "llm_feedback", False):
        return None

    metrics = {
        "test": _eval_split(logits, test_x, test_y),
        "val": _eval_split(logits, val_x, val_y),
    }
    training_summary = _summarize_training_trace(training_trace)
    params = {k: _to_float(v) for k, v in cf.get_model_conf().items()}
    payload = {
        "task": "llm_guided_training_diagnosis_auto_tuning",
        "dataset": cf.dataset,
        "metric_focus": ["macro_f1", "auprc"],
        "best_epoch": None if stopper is None else int(stopper.best_epoch),
        "params": params,
        "loss_weight": [float(loss_weight[0]), float(loss_weight[1])],
        "data_profile": _to_float(data_profile or {}),
        "label_profile": {
            "val": _label_profile("val", val_y),
            "test": _label_profile("test", test_y),
        },
        "metrics": metrics,
        "training_summary": training_summary,
        "runtime_profile": _to_float(runtime_profile or {}),
        "fusion_attention_weights": {k: _to_float(v) for k, v in best_weights.items()},
        "dap_gate_values": _collect_dap_gates(model),
        "allowed_feedback_targets": [
            "dap_gate_init",
            "loss_w0",
            "loss_w1",
        ],
        "feedback_constraints": {
            "auto_applied_targets": ["dap_gate_init", "loss_w0", "loss_w1"],
            "max_abs_change": {
                "dap_gate_init": 0.5,
                "loss_w0": 0.02,
                "loss_w1": 0.02,
            },
            "fixed_targets": ["dap_bins", "dap_temp", "dap_dropout"],
        },
        "frozen_components": [
            "feature graph generation",
            "semantic graph generation",
            "original graph attention fusion",
            "GCN",
            "RWR and BiLSTM aggregation",
            "classifier architecture",
        ],
    }
    payload["diagnosis_tags"] = _diagnose(payload)

    out_dir = getattr(args, "llm_feedback_dir", "results/llm_feedback")
    os.makedirs(out_dir, exist_ok=True)
    prefix = f"{cf.dataset}_{int(time.time())}"
    payload_path = os.path.join(out_dir, f"{prefix}_payload.json")
    feedback_path = os.path.join(out_dir, f"{prefix}_feedback.json")
    prompt_path = os.path.join(out_dir, f"{prefix}_prompt.txt")

    prompt = _build_prompt(payload)
    with open(payload_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    with open(prompt_path, "w", encoding="utf-8") as f:
        f.write(prompt)

    provider = getattr(args, "llm_provider", "qwen")
    defaults = _provider_defaults(provider)
    model_name = getattr(args, "llm_model", None) or defaults["model"]
    base_url = getattr(args, "llm_base_url", None) or defaults["base_url"]
    api_key_env = getattr(args, "llm_api_key_env", None) or defaults["api_key_env"]
    feedback = None
    if not getattr(args, "llm_feedback_dry_run", False):
        feedback = _call_llm(prompt, provider, model_name, base_url, api_key_env)
    rule_feedback = _rule_feedback(payload)
    rule_feedback["provider"] = provider
    rule_feedback["model"] = model_name
    rule_feedback["api_key_env"] = api_key_env

    if feedback is None or "error" in feedback:
        if feedback is not None and "error" in feedback:
            rule_feedback["llm_error"] = feedback["error"]
        feedback = rule_feedback
    else:
        feedback = {
            "provider": provider,
            "model": model_name,
            "api_key_env": api_key_env,
            "llm_diagnosis_text": feedback.get("text", ""),
            "llm_diagnosis_parsed": feedback.get("parsed"),
            "diagnosis_tags": payload["diagnosis_tags"],
            "next_run_suggestions": rule_feedback.get("next_run_suggestions", []),
            "auto_tuning_policy": (
                "Qwen is used for training diagnosis only. Stage-2 parameters are generated "
                "by bounded deterministic rules from diagnosis tags."
            ),
            "paper_description": rule_feedback.get("paper_description", ""),
        }

    with open(feedback_path, "w", encoding="utf-8") as f:
        json.dump(feedback, f, ensure_ascii=False, indent=2)

    print("LLM feedback payload:", payload_path)
    print("LLM feedback prompt:", prompt_path)
    print("LLM feedback output:", feedback_path)
    print("LLM feedback summary:", feedback)
    return feedback
