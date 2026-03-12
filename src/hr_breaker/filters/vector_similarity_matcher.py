from litellm import aembedding as litellm_aembedding

from hr_breaker.config import get_embedding_dimensions, get_embedding_request_kwargs, get_settings
from hr_breaker.filters.base import BaseFilter
from hr_breaker.filters.registry import FilterRegistry
from hr_breaker.models import FilterResult, JobPosting, OptimizedResume, ResumeSource
from hr_breaker.models.language import Language
from hr_breaker.utils.retry import run_with_retry


@FilterRegistry.register
class VectorSimilarityMatcher(BaseFilter):
    """Vector similarity filter using embeddings via litellm."""

    name = "VectorSimilarityMatcher"
    priority = 6

    @property
    def threshold(self) -> float:
        return get_settings().filter_vector_threshold

    async def evaluate(
        self,
        optimized: OptimizedResume,
        job: JobPosting,
        source: ResumeSource,
        language: Language | None = None,
    ) -> FilterResult:
        if optimized.pdf_text is None:
            return FilterResult(
                filter_name=self.name,
                passed=False,
                score=0.0,
                threshold=self.threshold,
                issues=["No PDF text available"],
                suggestions=["Ensure PDF compilation succeeds"],
            )

        resume_text = optimized.pdf_text
        job_text = f"{job.title} {job.description} {' '.join(job.requirements)}"

        dimensions = get_embedding_dimensions()

        try:
            if dimensions is None:
                result = await run_with_retry(
                    litellm_aembedding,
                    **get_embedding_request_kwargs(),
                    input=[resume_text, job_text],
                )
            else:
                result = await run_with_retry(
                    litellm_aembedding,
                    **get_embedding_request_kwargs(),
                    input=[resume_text, job_text],
                    dimensions=dimensions,
                )
            embeddings = [item["embedding"] for item in result.data]
        except Exception as e:
            return FilterResult(
                filter_name=self.name,
                passed=True,
                score=0.5,
                threshold=self.threshold,
                issues=[f"Embedding API unavailable, filter skipped: {e}"],
                suggestions=["Check embedding API key and model configuration"],
            )

        # Cosine similarity
        e1, e2 = embeddings[0], embeddings[1]
        dot = sum(a * b for a, b in zip(e1, e2))
        norm1 = sum(a * a for a in e1) ** 0.5
        norm2 = sum(b * b for b in e2) ** 0.5
        similarity = dot / (norm1 * norm2) if norm1 and norm2 else 0.0

        # Normalize to 0-1 (cosine similarity is -1 to 1)
        score = (similarity + 1) / 2

        issues = []
        if score < self.threshold:
            issues.append(
                f"Low semantic vector similarity to job posting ({score:.2f})"
            )

        return FilterResult(
            filter_name=self.name,
            passed=score >= self.threshold,
            score=score,
            threshold=self.threshold,
            issues=issues,
            suggestions=[],
        )
