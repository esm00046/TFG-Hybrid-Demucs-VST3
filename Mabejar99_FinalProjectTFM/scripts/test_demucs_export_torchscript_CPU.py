import time
from pathlib import Path

import torch


def main():
    # ============================================================
    # 1. Seleccionar dispositivo CPU
    # ============================================================

    print("PyTorch:", torch.__version__)

    device = torch.device("cpu")

    print("Usando dispositivo:", device)

    # ============================================================
    # 2. Ruta del modelo exportado
    # ============================================================

    PROJECT_ROOT = Path(__file__).resolve().parents[1]

    model_path = PROJECT_ROOT / "exported_models" / "demucs_export_torchscript_CPU.pt"

    print("\nRuta del modelo TorchScript:")
    print(model_path)

    if not model_path.exists():
        raise FileNotFoundError(f"No se ha encontrado el modelo: {model_path}")

    # ============================================================
    # 3. Cargar modelo TorchScript
    # ============================================================

    print("\nCargando modelo TorchScript en CPU...")

    model = torch.jit.load(str(model_path), map_location="cpu")
    model.eval()

    # Forzamos float32.
    model = model.to(dtype=torch.float32)

    # Lo dejamos explícitamente en CPU.
    model = model.to(device)

    print("Modelo cargado correctamente.")

    # ============================================================
    # 4. Comprobar parámetros
    # ============================================================

    parametro_encontrado = False

    for name, param in model.named_parameters():
        print("\nPrimer parámetro encontrado:")
        print("Nombre:", name)
        print("Dispositivo:", param.device)
        print("Tipo:", param.dtype)
        parametro_encontrado = True
        break

    if not parametro_encontrado:
        print("\nNo se han encontrado parámetros visibles en el modelo.")

    # ============================================================
    # 5. Crear entrada de prueba
    # ============================================================
    #
    # Mismo formato usado en la exportación:
    #
    #     [1, 12, 8192]
    #
    # batch = 1
    # canales = 12
    # muestras = 8192
    #

    num_channels = 12
    num_samples = 8192

    input_tensor = torch.randn(
        1,
        num_channels,
        num_samples,
        dtype=torch.float32,
        device=device
    )

    print("\nEntrada de prueba:")
    print("Forma:", input_tensor.shape)
    print("Tipo:", input_tensor.dtype)
    print("Dispositivo:", input_tensor.device)

    # ============================================================
    # 6. Pasada de calentamiento
    # ============================================================

    print("\nEjecutando pasada de calentamiento...")

    with torch.inference_mode():
        output_tensor = model(input_tensor)

    print("Calentamiento completado.")

    # ============================================================
    # 7. Medir tiempo de inferencia
    # ============================================================

    print("\nMidiendo tiempo de inferencia...")

    with torch.inference_mode():
        inicio = time.perf_counter()

        output_tensor = model(input_tensor)

        fin = time.perf_counter()

    tiempo_ms = (fin - inicio) * 1000.0

    # ============================================================
    # 8. Mostrar resultados
    # ============================================================

    print("\nResultados:")
    print("Forma de salida:", output_tensor.shape)
    print("Tipo salida:", output_tensor.dtype)
    print("Dispositivo salida:", output_tensor.device)
    print("Tiempo de inferencia:", tiempo_ms, "ms")

    print("Output min:", output_tensor.min().item())
    print("Output max:", output_tensor.max().item())
    print("Output abs max:", output_tensor.abs().max().item())

    if output_tensor.ndim == 3:
        print("\nInterpretación de salida:")
        print("Batch:", output_tensor.shape[0])
        print("Canales/salidas:", output_tensor.shape[1])
        print("Muestras:", output_tensor.shape[2])

        if output_tensor.shape[1] == 12:
            print("Correcto: el modelo devuelve 12 salidas.")
        else:
            print("AVISO: el número de salidas no es 12.")

        if output_tensor.shape[2] == num_samples:
            print("Correcto: la salida tiene el mismo número de muestras que la entrada.")
        else:
            print("AVISO: la salida no tiene el mismo número de muestras que la entrada.")
            print("Entrada muestras:", num_samples)
            print("Salida muestras:", output_tensor.shape[2])
    else:
        print("AVISO: la salida no tiene 3 dimensiones.")


if __name__ == "__main__":
    main()