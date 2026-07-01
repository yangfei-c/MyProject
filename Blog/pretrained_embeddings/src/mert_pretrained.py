from __future__ import annotations

from time import perf_counter
import numpy as np
import torch
from transformers import AutoModel, Wav2Vec2FeatureExtractor

from audio_process import resample_audio
class MertExtractor:
    def __init__(
        self,
        model_id: str,
        device: torch.device | str,
    ) -> None:
        self.model_name = "mert"
        self.model_id = model_id
        self.device = torch.device(device) if not isinstance(device, torch.device) else device

        self.processor = Wav2Vec2FeatureExtractor.from_pretrained(
            model_id,
        )
        self.target_sample_rate = int(getattr(self.processor, "sampling_rate"))
        self.model = AutoModel.from_pretrained(
            model_id,
            trust_remote_code=True,
        ).to(self.device)
        self.model.eval()

    def _forward_segment(self, segment: np.ndarray, source_sample_rate: int) -> np.ndarray:
        audio = resample_audio(
            segment,
            original_sample_rate=source_sample_rate,
            target_sample_rate=self.target_sample_rate,
        )
        inputs = self.processor(audio, sampling_rate=self.target_sample_rate, return_tensors="pt")
        inputs = {key: value.to(self.device) for key, value in inputs.items()}
        with torch.inference_mode():
            outputs = self.model(**inputs, output_hidden_states=True)
        hidden_states = outputs.hidden_states[-1] if outputs.hidden_states else outputs.last_hidden_state
        pooled = hidden_states.mean(dim=1)
        return pooled.squeeze(0).detach().cpu().numpy().astype(np.float32, copy=False)

    def extract(
        self,
        segments: list[np.ndarray],
        original_sample_rate: int,
    ) -> dict[str, object]:
        if not segments:
            raise ValueError("No valid audio segments were provided")

        started_at = perf_counter()
        segment_embeddings = np.stack(
            [self._forward_segment(segment, int(original_sample_rate)) for segment in segments],
            axis=0,
        ).astype(np.float32, copy=False)
        track_embedding = segment_embeddings.mean(axis=0).astype(np.float32, copy=False)
        runtime_seconds = perf_counter() - started_at

        return {
            "model_name": self.model_name,
            "model_id": self.model_id,
            "target_sample_rate": self.target_sample_rate,
            "segment_embeddings": segment_embeddings,
            "track_embedding": track_embedding,
            "embedding_dim": int(segment_embeddings.shape[1]),
            "runtime_seconds": runtime_seconds,
        }
