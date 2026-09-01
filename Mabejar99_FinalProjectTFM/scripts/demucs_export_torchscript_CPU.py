import os
import sys
from pathlib import Path

import torch
import yaml

# ============================================================
# Localizar carpeta raíz del proyecto
# ============================================================
#
# Este script está dentro de:
#
#     Mabejar99_FinalProjectTFM/scripts/
#
# Por tanto, parents[1] apunta a:
#
#     Mabejar99_FinalProjectTFM/
#
# Ahí están:
#     conf.yaml
#     checkpoints/
#     demucs/
#     exported_models/
#

PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Añadimos la raíz del proyecto al path para poder importar:
#
#     from demucs.hdemucs import HDemucs
#
sys.path.insert(0, str(PROJECT_ROOT))

from demucs.hdemucs import HDemucs


def main():
    # ============================================================
    # 1. Seleccionar dispositivo CPU
    # ============================================================

    print("PyTorch:", torch.__version__)

    device = torch.device("cpu")

    print("Usando dispositivo:", device)

    # ============================================================
    # 2. Rutas
    # ============================================================

    config_path = PROJECT_ROOT / "conf.yaml"

    model_path = PROJECT_ROOT / "checkpoints" / "best_epoch=155.ckpt"

    output_folder = PROJECT_ROOT / "exported_models"
    output_folder.mkdir(parents=True, exist_ok=True)

    output_path = output_folder / "demucs_export_torchscript_CPU.pt"

    print("\nRutas:")
    print("Config:", config_path)
    print("Checkpoint:", model_path)
    print("Salida:", output_path)

    # ============================================================
    # 3. Leer configuración del modelo
    # ============================================================

    with open(config_path, "r") as f:
        conf = yaml.safe_load(f)

    S = conf["separator_conf"]["n_srcs"]

    confdemucs = {
        "sources": [str(i) for i in range(S)],
        "audio_channels": conf["separator_conf"]["n_imics"],
        "cac": conf["hdemucs_conf"]["cac"],
        "samplerate": conf["hdemucs_conf"]["samplerate"],
        "channels": conf["hdemucs_conf"]["channels"],
        "segment": conf["dataset"]["chunk_duration"],
    }

    print("\nConfiguración usada para crear HDemucs:")
    print("sources:", confdemucs["sources"])
    print("audio_channels:", confdemucs["audio_channels"])
    print("cac:", confdemucs["cac"])
    print("samplerate:", confdemucs["samplerate"])
    print("channels:", confdemucs["channels"])
    print("segment:", confdemucs["segment"])

    # ============================================================
    # 4. Crear modelo
    # ============================================================

    model = HDemucs(**confdemucs)

    # ============================================================
    # 5. Cargar checkpoint
    # ============================================================
    #
    # El checkpoint viene del entrenamiento y contiene un state_dict.
    # Igual que en el cuaderno, eliminamos:
    #
    #   - el prefijo "model."
    #   - las claves que empiezan por "auralossnew."
    #

    checkpoint = torch.load(model_path, map_location="cpu")

    state_dict = {
        k.replace("model.", ""): v
        for k, v in checkpoint["state_dict"].items()
    }

    state_dict = {
        k: v
        for k, v in state_dict.items()
        if not k.startswith("auralossnew.")
    }

    model.load_state_dict(state_dict)

    # Modo inferencia.
    model.eval()

    # Forzamos float32.
    model = model.to(dtype=torch.float32)

    # Movemos explícitamente a CPU.
    model = model.to(device)

    print("\nModelo cargado correctamente.")

    # ============================================================
    # 6. Crear entrada de ejemplo EN CPU
    # ============================================================
    #
    # La forma usada será la misma que en MPS:
    #
    #     [1, 12, 8192]
    #
    # Es decir:
    #     batch = 1
    #     canales = 12
    #     muestras = 8192
    #

    audio_channels = confdemucs["audio_channels"]

    example_num_samples = 8192

    example_input = torch.randn(
        1,
        audio_channels,
        example_num_samples,
        dtype=torch.float32,
        device=device
    )

    print("\nEntrada de ejemplo:")
    print("Dispositivo example_input:", example_input.device)
    print("Tipo example_input:", example_input.dtype)
    print("Forma example_input:", example_input.shape)

    # Comprobamos que el modelo está en CPU.
    for name, param in model.named_parameters():
        print("\nPrimer parámetro:", name)
        print("Dispositivo parámetro:", param.device)
        print("Tipo parámetro:", param.dtype)
        break

    # ============================================================
    # 7. Probar forward antes de exportar
    # ============================================================

    print("\nProbando forward en CPU antes de exportar...")

    with torch.inference_mode():
        output = model(example_input)

    print("Forward completado correctamente.")
    print("Dispositivo salida:", output.device)
    print("Tipo salida:", output.dtype)
    print("Forma salida:", output.shape)
    print("Output abs max:", output.abs().max().item())

    # ============================================================
    # 8. Hacer trace en CPU
    # ============================================================

    print("\nExportando modelo TorchScript en CPU...")

    with torch.inference_mode():
        traced_model = torch.jit.trace(
            model,
            example_input,
            strict=False
        )

    # ============================================================
    # 9. Guardar modelo
    # ============================================================

    traced_model.save(str(output_path))

    print("\nModelo exportado correctamente.")
    print("Guardado en:", output_path)


if __name__ == "__main__":
    main()