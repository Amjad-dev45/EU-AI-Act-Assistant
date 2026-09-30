"""Optional RAGAS scoring; imports are lazy so the application needs no RAGAS."""
import asyncio
import math
import time

METRICS = ("faithfulness", "answer_relevancy", "context_precision", "context_recall", "factual_correctness")


def metric_inputs(sample):
    common = {"user_input": sample["user_input"]}
    ranked = {**common, "reference": sample["reference"], "retrieved_contexts": sample["retrieved_contexts"]}
    return {
        "faithfulness": {**common, "response": sample["response"], "retrieved_contexts": sample["generation_contexts"]},
        "answer_relevancy": {**common, "response": sample["response"]},
        "context_precision": ranked, "context_recall": ranked,
        "factual_correctness": {"response": sample["response"], "reference": sample["reference"]},
    }


def make_scorers(client, model, embedding_model):
    from ragas.llms import llm_factory
    from ragas.embeddings.base import embedding_factory
    from ragas.metrics.collections import Faithfulness, AnswerRelevancy, ContextPrecision, ContextRecall, FactualCorrectness
    llm = llm_factory(model, client=client, temperature=0, max_tokens=4096, max_retries=1)
    embeddings = embedding_factory("openai", model=embedding_model, client=client)
    return {"faithfulness": Faithfulness(llm=llm),
            "answer_relevancy": AnswerRelevancy(llm=llm, embeddings=embeddings),
            "context_precision": ContextPrecision(llm=llm), "context_recall": ContextRecall(llm=llm),
            "factual_correctness": FactualCorrectness(llm=llm, mode="f1")}


async def judge_sample(sample, scorers, timeout=120):
    result = {**sample, "scores": {}, "errors": {}}
    if sample.get("skip_reason"):
        return result
    started = time.perf_counter()
    inputs = metric_inputs(sample)
    for name, scorer in scorers.items():
        try:
            score = await asyncio.wait_for(scorer.ascore(**inputs[name]), timeout)
            value = float(score.value)
            if not math.isfinite(value):
                raise ValueError("Metric was undefined.")
            result["scores"][name] = value
        except Exception as error:
            # A timeout/undefined score is not a zero and must not lower the mean.
            result["scores"][name] = None
            result["errors"][name] = type(error).__name__
    result["elapsed_seconds"] = round(time.perf_counter() - started, 3)
    return result


def summarize(rows):
    metrics = {}
    for name in METRICS:
        values = [r.get("scores", {}).get(name) for r in rows]
        scored = [v for v in values if v is not None]
        metrics[name] = {"mean": sum(scored) / len(scored) if scored else None,
                         "scored": len(scored), "unscored": len(rows) - len(scored)}
    return {"selected": len(rows), "skipped": sum(bool(r.get("skip_reason")) for r in rows),
            "rows_with_errors": sum(bool(r.get("errors")) for r in rows), "metrics": metrics}
