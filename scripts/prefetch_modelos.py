"""Descarga todos los modelos durante el build de la imagen Docker.

Ejecutar en la raiz del proyecto (con seed-vc/ presente y torch instalado).
Solo descarga; no instancia los modelos, para que el build no use RAM de mas.
"""
from pathlib import Path

CHECKPOINTS = "./checkpoints"


def main() -> None:
    from huggingface_hub import hf_hub_download, snapshot_download

    raiz = Path(__file__).resolve().parents[1]
    if not (raiz / "seed-vc" / "seed_vc_wrapper.py").exists():
        raise SystemExit("Falta seed-vc/ junto a la raiz del proyecto.")

    # Seed-VC: el cargador propio (hf_utils) usa cache_dir="./checkpoints"
    for nombre in (
        "DiT_seed_v2_uvit_whisper_small_wavenet_bigvgan_pruned.pth",
        "config_dit_mel_seed_uvit_whisper_small_wavenet.yml",
        "DiT_seed_v2_uvit_whisper_base_f0_44k_bigvgan_pruned_ft_ema.pth",
        "config_dit_mel_seed_uvit_whisper_base_f0_44k.yml",
    ):
        print("Plachta/Seed-VC:", nombre, flush=True)
        hf_hub_download(repo_id="Plachta/Seed-VC", filename=nombre,
                        cache_dir=CHECKPOINTS)

    print("funasr/campplus", flush=True)
    hf_hub_download(repo_id="funasr/campplus",
                    filename="campplus_cn_common.bin", cache_dir=CHECKPOINTS)

    print("lj1995/VoiceConversionWebUI: rmvpe.pt", flush=True)
    hf_hub_download(repo_id="lj1995/VoiceConversionWebUI", filename="rmvpe.pt",
                    cache_dir=CHECKPOINTS)

    # Vocoders BigVGAN (cache normal de huggingface_hub)
    for repo in ("nvidia/bigvgan_v2_22khz_80band_256x",
                 "nvidia/bigvgan_v2_44khz_128band_512x"):
        print(repo, flush=True)
        snapshot_download(repo_id=repo,
                          allow_patterns=["*.json", "*.pth", "*.pt", "*.yml"])

    # Tokenizador de habla (whisper) y modelo de voz hablada
    print("openai/whisper-small", flush=True)
    snapshot_download(repo_id="openai/whisper-small")
    print("resemble-ai/chatterbox-multilingual-tts", flush=True)
    snapshot_download(repo_id="resemble-ai/chatterbox-multilingual-tts")

    # Pesos de Demucs (cache de torch.hub)
    import torch
    print("demucs htdemucs", flush=True)
    torch.hub.load("demucs/demucs", "htdemucs", source="github",
                   trust_repo=True)

    print("Modelos descargados correctamente.", flush=True)


if __name__ == "__main__":
    main()
