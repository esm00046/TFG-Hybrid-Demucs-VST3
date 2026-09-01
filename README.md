# TFG-Hybrid-Demucs-VST3

Repositorio del Trabajo Fin de Grado:

**De-bleeding en tiempo real de micrófonos cercanos mediante IA: diseño e implementación de un VST3 multicanal**

Este proyecto desarrolla un prototipo de complemento de audio **VST3 multicanal** capaz de ejecutar un modelo basado en **Hybrid Demucs** dentro de **REAPER** para reducir el *microphone bleeding* en grabaciones realizadas con micrófonos cercanos.

El sistema parte de un modelo previamente entrenado para trabajar con señales de hasta doce canales. La contribución principal de este TFG consiste en adaptar dicho modelo a un flujo de procesamiento de audio en tiempo real mediante **C++**, **JUCE**, **LibTorch**, **TorchScript** y **REAPER**.

---

## 1. Descripción general

En una grabación musical multicanal es habitual colocar un micrófono cerca de cada instrumento o fuente sonora. Sin embargo, cada micrófono puede captar también parte del sonido procedente de las demás fuentes. Este fenómeno se conoce como **microphone bleeding**, *leakage* o *spill*, y reduce la independencia entre las pistas durante la mezcla.

El objetivo de este proyecto es acercar un modelo de separación de fuentes basado en inteligencia artificial a un flujo de trabajo real dentro de una estación de trabajo de audio digital. Para ello, se ha desarrollado un complemento VST3 que puede insertarse directamente en REAPER y procesar audio multicanal mediante un modelo Hybrid Demucs exportado a TorchScript.

La solución desarrollada permite:

- Cargar un modelo Hybrid Demucs exportado a TorchScript.
- Ejecutar el modelo desde C++ mediante LibTorch.
- Procesar audio dentro de un complemento VST3 desarrollado con JUCE.
- Trabajar con configuraciones de entre 1 y 12 canales reales.
- Rellenar internamente con ceros los canales no utilizados hasta completar los 12 canales esperados por el modelo.
- Utilizar una ventana histórica de 8192 muestras.
- Aplicar solapamiento-suma de 512 muestras para mejorar la continuidad entre bloques.
- Ejecutar la inferencia mediante CPU o mediante MPS/Apple Metal.
- Automatizar el enrutamiento multicanal en REAPER mediante ReaScript.
- Crear un bus multicanal y pistas separadas de salida para trabajar con los canales procesados.

---

## 2. Arquitectura de la solución

La arquitectura general del sistema puede resumirse así:

```text
Pistas de audio en REAPER
        ↓
ReaScript de automatización
        ↓
Bus multicanal Demucs12
        ↓
Plugin VST3 Demucs12
        ↓
JUCE AudioBuffer
        ↓
Tensor [1, 12, 8192]
        ↓
Modelo Hybrid Demucs exportado a TorchScript
        ↓
Tensor de salida [1, 12, 8192]
        ↓
Reconstrucción mediante solapamiento-suma
        ↓
Pistas separadas de salida en REAPER
```

El complemento no entrena el modelo ni modifica su arquitectura interna. Su función es cargar el modelo previamente exportado, adaptar los bloques de audio recibidos desde REAPER a la entrada esperada por la red y devolver las señales procesadas manteniendo la correspondencia entre canales.

---

## 3. Contenido del repositorio

La organización general del repositorio es la siguiente:

```text
TFG-Hybrid-Demucs-VST3/
│
├── Mabejar99_FinalProjectTFM/
│   ├── conf.yaml
│   ├── demucs/
│   ├── checkpoints/
│   ├── audio/
│   ├── scripts/
│   └── exported_models/
│
├── Plugins/
│   └── Demucs12/
│       ├── Demucs12.jucer
│       ├── Source/
│       │   ├── PluginProcessor.h
│       │   ├── PluginProcessor.cpp
│       │   ├── PluginEditor.h
│       │   └── PluginEditor.cpp
│       └── Builds/
│
├── cpp_offline_tests/
│
├── scripts/
│   └── Demucs12_Bus.lua
│
├── docs/
│    └── Memoria_TFG.pdf
│
└── README.md
```

> **Nota:** la estructura puede variar ligeramente en función de cómo se haya organizado la copia final del repositorio. En cualquier caso, las carpetas principales corresponden al modelo heredado, al complemento JUCE/VST3, a las pruebas offline en C++, a los scripts de REAPER y a la documentación.

---

## 4. Carpeta del modelo heredado

La carpeta:

```text
Mabejar99_FinalProjectTFM/
```

contiene el modelo heredado basado en Hybrid Demucs, su configuración y los scripts necesarios para exportarlo y validarlo.

Elementos principales:

- `conf.yaml`: archivo de configuración del modelo.
- `demucs/`: implementación de Hybrid Demucs.
- `checkpoints/`: carpeta destinada al checkpoint entrenado.
- `audio/`: señales de audio utilizadas durante las pruebas.
- `scripts/`: scripts de exportación y validación.
- `exported_models/`: carpeta donde se almacenan los modelos TorchScript exportados.

El modelo definitivo corresponde a una adaptación multicanal de Hybrid Demucs implementada mediante la clase `HDemucs`, definida dentro del archivo:

```text
demucs/hdemucs.py
```

La configuración se obtiene desde:

```text
conf.yaml
```

y los parámetros entrenados se cargan desde el checkpoint:

```text
checkpoints/best_epoch=155.ckpt
```

Este checkpoint procede del trabajo previo y no se vuelve a entrenar en este TFG.

---

## 5. Carpeta del complemento VST3

La carpeta del complemento contiene el proyecto JUCE utilizado para generar el VST3.

Archivos principales:

- `PluginProcessor.h`: declaración de la clase principal del procesador.
- `PluginProcessor.cpp`: implementación del procesamiento de audio, carga del modelo, inferencia y reconstrucción de salida.
- `PluginEditor.h`: declaración de la interfaz gráfica.
- `PluginEditor.cpp`: implementación de la interfaz informativa.
- `Demucs12.jucer`: proyecto de Projucer.

La clase principal del complemento se encarga de:

- Configurar los buses de entrada y salida.
- Validar configuraciones de entre 1 y 12 canales reales.
- Cargar el modelo TorchScript mediante LibTorch.
- Recibir bloques de audio desde REAPER.
- Mantener una ventana histórica de entrada.
- Construir tensores con dimensiones `[1, 12, 8192]`.
- Ejecutar la inferencia.
- Reconstruir la salida correspondiente.
- Aplicar solapamiento-suma.
- Devolver el audio procesado al anfitrión.

La interfaz gráfica del plugin tiene una función únicamente informativa. No contiene parámetros modificables por el usuario, ya que el flujo de trabajo práctico se gestiona mediante REAPER y ReaScript.

---

## 6. Pruebas offline en C++

La carpeta:

```text
cpp_offline_tests/
```

contiene proyectos de prueba utilizados para validar la carga y ejecución del modelo desde C++ fuera de REAPER.

Estas pruebas permiten comprobar que:

- LibTorch está correctamente enlazado.
- El archivo `.pt` puede cargarse desde C++.
- El modelo acepta una entrada con dimensiones `[1, 12, 8192]`.
- La inferencia devuelve una salida con la forma esperada.
- El problema no procede de REAPER ni de JUCE si la carga del modelo falla.

Estas pruebas fueron útiles antes de integrar el modelo en el complemento VST3.

---

## 7. ReaScript de automatización

La carpeta:

```text
scripts/
```

contiene el ReaScript utilizado para automatizar el flujo de trabajo dentro de REAPER.

Archivo principal:

```text
Demucs12_Bus.lua
```

Este script:

1. Lee las pistas seleccionadas en REAPER.
2. Calcula el número de canales reales.
3. Crea un bus multicanal.
4. Configura el número de canales del bus.
5. Inserta el plugin Demucs12 en el bus.
6. Crea pistas separadas de salida.
7. Redirige cada canal del bus a su pista correspondiente.
8. Desactiva el envío al máster de las pistas originales.
9. Desactiva el envío al máster del bus.

Su finalidad es evitar que el usuario tenga que configurar manualmente el ruteo multicanal de doce canales dentro de REAPER.

---

## 8. Documentación

La carpeta:

```text
docs/
```

puede incluir:

- Memoria del TFG.
- Capturas de configuración.
- Evidencias de ejecución.
- Capturas de tiempos de inferencia.
- Documentación complementaria.
- Material adicional utilizado durante la validación.

---

## 9. Requisitos

La configuración empleada durante el desarrollo fue:

- macOS.
- Equipo Apple Silicon.
- Procesador Apple M4.
- Xcode.
- JUCE 8.
- Projucer.
- REAPER.
- Miniconda o Anaconda.
- Python.
- PyTorch.
- TorchScript.
- LibTorch para macOS ARM64.
- Soporte MPS para aceleración mediante Apple Metal.

El desarrollo final se validó principalmente en macOS con MPS. No se utilizó CUDA, ya que CUDA está orientado a GPU NVIDIA y no está disponible en el equipo Apple utilizado.

---

## 10. Preparación del entorno Python

Primero se debe clonar el repositorio:

```bash
git clone https://github.com/esm00046/TFG-Hybrid-Demucs-VST3.git
cd TFG-Hybrid-Demucs-VST3
```

Después se accede a la carpeta del modelo:

```bash
cd Mabejar99_FinalProjectTFM
```

El entorno original del modelo puede contener dependencias relacionadas con CUDA. En equipos Apple Silicon estas dependencias deben eliminarse o adaptarse, ya que CUDA no se utiliza en macOS con GPU Apple.

Una vez preparado el archivo de entorno, se crea el entorno de Python:

```bash
conda env create -f environment-cuda.yml
```

Después se activa:

```bash
conda activate demucs
```

Si se van a ejecutar notebooks o scripts desde Visual Studio Code, puede ser necesario instalar algunos paquetes adicionales:

```bash
pip install ipython ipywidgets
```

> **Nota:** el nombre del entorno y del archivo `.yml` puede variar según la versión incluida en el repositorio. Si el archivo tiene otro nombre, debe sustituirse en los comandos anteriores.

---

## 11. Archivos pesados y modelos entrenados

El repositorio puede no incluir archivos pesados como checkpoints o modelos exportados, dependiendo de la política de publicación utilizada.

Archivos importantes:

```text
Mabejar99_FinalProjectTFM/checkpoints/best_epoch=155.ckpt
Mabejar99_FinalProjectTFM/exported_models/demucs_export_torchscript_CPU.pt
Mabejar99_FinalProjectTFM/exported_models/demucs_export_torchscript_MPS.pt
```

Si estos archivos no están incluidos en el repositorio, deben colocarse manualmente en sus carpetas correspondientes.

El archivo:

```text
best_epoch=155.ckpt
```

contiene los parámetros entrenados del modelo heredado.

A partir de él se generan los módulos TorchScript:

```text
demucs_export_torchscript_CPU.pt
demucs_export_torchscript_MPS.pt
```

Estos archivos `.pt` son los que posteriormente carga el complemento VST3 desde C++ mediante LibTorch.

---

## 12. Exportación del modelo a TorchScript

El complemento no carga directamente el archivo `.ckpt`. Para poder ejecutar el modelo desde C++ mediante LibTorch, primero hay que exportarlo a TorchScript.

Desde la carpeta:

```bash
cd Mabejar99_FinalProjectTFM
```

### Exportación para CPU

```bash
python scripts/demucs_export_torchscript_CPU.py
```

Salida esperada:

```text
exported_models/demucs_export_torchscript_CPU.pt
```

### Exportación para MPS / Apple Metal

```bash
python scripts/demucs_export_torchscript_MPS.py
```

Salida esperada:

```text
exported_models/demucs_export_torchscript_MPS.pt
```

Los scripts reconstruyen la arquitectura Hybrid Demucs, cargan el checkpoint entrenado, eliminan los elementos no necesarios para inferencia y generan una representación TorchScript mediante `torch.jit.trace()`.

La entrada de ejemplo utilizada durante la exportación tiene dimensiones:

```text
[1, 12, 8192]
```

Por este motivo, la versión final del complemento utiliza una ventana histórica de 8192 muestras.

---

## 13. Diferencia entre `.ckpt` y `.pt`

Durante el proyecto se utilizan dos tipos de archivos relacionados con el modelo:

### Archivo `.ckpt`

El archivo `.ckpt` contiene el estado del modelo obtenido durante el entrenamiento. Procede del trabajo previo y conserva los pesos entrenados junto con otra información asociada al entrenamiento.

Ejemplo:

```text
best_epoch=155.ckpt
```

### Archivo `.pt`

El archivo `.pt` contiene el modelo exportado mediante TorchScript. Este es el archivo preparado para inferencia y compatible con LibTorch desde C++.

Ejemplos:

```text
demucs_export_torchscript_CPU.pt
demucs_export_torchscript_MPS.pt
```

El complemento VST3 carga el archivo `.pt`, no el `.ckpt`.

---

## 14. Configuración de LibTorch

Para compilar el complemento es necesario descargar LibTorch para macOS ARM64 desde la página oficial de PyTorch.

La carpeta de LibTorch debe guardarse en una ubicación fija. Durante el desarrollo se utilizó una ruta local del tipo:

```text
/Users/<usuario>/Desktop/TFG/libtorch/
```

Si macOS bloquea las bibliotecas dinámicas por haber sido descargadas de Internet, puede eliminarse el atributo de cuarentena:

```bash
xattr -cr /ruta/a/libtorch
```

Ejemplo:

```bash
xattr -cr /Users/<usuario>/Desktop/TFG/libtorch
```

---

## 15. Configuración del proyecto JUCE

Abrir el proyecto del complemento desde Projucer:

```text
Plugins/Demucs12/Demucs12.jucer
```

En el exportador de Xcode deben configurarse las rutas de LibTorch.

### Header Search Paths

```text
/ruta/a/libtorch/include
/ruta/a/libtorch/include/torch/csrc/api/include
```

Ejemplo:

```text
/Users/<usuario>/Desktop/TFG/libtorch/include
/Users/<usuario>/Desktop/TFG/libtorch/include/torch/csrc/api/include
```

### Extra Library Search Paths

```text
/ruta/a/libtorch/lib
```

Ejemplo:

```text
/Users/<usuario>/Desktop/TFG/libtorch/lib
```

### External Libraries to Link

```text
torch
torch_cpu
c10
```

### Extra Linker Flags

```text
-Wl,-rpath,/ruta/a/libtorch/lib -mmacosx-version-min=11.0
```

Ejemplo:

```text
-Wl,-rpath,/Users/<usuario>/Desktop/TFG/libtorch/lib -mmacosx-version-min=11.0
```

### Extra Compiler Flags

```text
-fno-modules -mmacosx-version-min=11.0
```

### Estándar C++

El proyecto debe compilarse con:

```text
C++17
```

---

## 16. Ruta del modelo en el plugin

El complemento carga el modelo TorchScript desde una ruta definida en el código, dentro de la función `loadModel()` de `PluginProcessor.cpp`.

Si el repositorio se clona en otra ubicación, es necesario revisar y ajustar la ruta del modelo.

Ejemplo orientativo:

```cpp
std::string modelpath =
    "/ruta/al/proyecto/Mabejar99_FinalProjectTFM/exported_models/";
```

Después se añade el archivo correspondiente:

```text
demucs_export_torchscript_MPS.pt
```

o:

```text
demucs_export_torchscript_CPU.pt
```

La versión final se ha probado principalmente con MPS:

```cpp
modelDevice = torch::Device(torch::kMPS);
```

Si se desea probar CPU, debe cargarse el modelo exportado para CPU y seleccionar:

```cpp
modelDevice = torch::Device(torch::kCPU);
```

---

## 17. Compilación del complemento VST3

Una vez configurado el proyecto en Projucer:

1. Guardar el proyecto.
2. Abrir el proyecto generado en Xcode.
3. Seleccionar el esquema correspondiente.
4. Compilar y generar el archivo `.vst3`.

El archivo VST3 debe instalarse en la carpeta de plugins de macOS:

```text
~/Library/Audio/Plug-Ins/VST3/
```

Si se desea copiar manualmente:

```bash
cp -R /ruta/al/plugin/Demucs12.vst3 ~/Library/Audio/Plug-Ins/VST3/
```

Después, abrir REAPER y realizar un escaneo de plugins.

---

## 18. Configuración de REAPER

En REAPER se recomienda trabajar con:

```text
Sample rate: 48000 Hz
Block size: 4096 samples
```

La frecuencia de 48 kHz coincide con la configuración utilizada por el modelo Hybrid Demucs.

El tamaño de bloque de 4096 muestras proporciona aproximadamente:

```text
4096 / 48000 = 0.0853 s
```

es decir, unos 85,3 ms de audio por bloque.

La configuración final del complemento utiliza:

```text
Ventana histórica: 8192 muestras
Solapamiento: 512 muestras
Canales internos: 12
```

---

## 19. Uso manual del plugin en REAPER

El plugin puede insertarse manualmente en una pista o bus de REAPER.

Pasos básicos:

1. Abrir REAPER.
2. Crear una pista.
3. Configurar la pista con el número de canales deseado.
4. Insertar el plugin:

```text
VST3: Demucs12
```

5. Reproducir el proyecto.

No obstante, para trabajar cómodamente con las salidas multicanal se recomienda utilizar el ReaScript incluido en el repositorio.

---

## 20. Uso recomendado mediante ReaScript

El flujo recomendado es utilizar el script:

```text
scripts/Demucs12_Bus.lua
```

Este script automatiza el enrutamiento necesario para usar el plugin dentro de REAPER.

### Pasos

1. Importar en REAPER una pista multicanal o varias pistas independientes.
2. Seleccionar las pistas que se quieren procesar.
3. Ir al menú:

```text
Actions -> Show action list
```

4. Buscar:

```text
ReaScript: Run ReaScript
```

5. Seleccionar el archivo:

```text
Demucs12_Bus.lua
```

6. Ejecutar el script.

El script creará automáticamente:

```text
Demucs12 Bus
Separated 1
Separated 2
Separated 3
...
Separated 12
```

El bus recibirá las señales seleccionadas, insertará el plugin Demucs12 y enviará cada canal de salida a una pista separada.

---

## 21. Funcionamiento del ReaScript

El ReaScript realiza las siguientes operaciones:

1. Comprueba las pistas seleccionadas.
2. Calcula el número de canales reales.
3. Crea un bus multicanal.
4. Configura el número de canales del bus.
5. Inserta el plugin Demucs12 en el bus.
6. Crea una pista de salida por cada canal.
7. Envía cada canal de salida del bus a una pista independiente.
8. Desactiva el envío al máster de las pistas originales.
9. Desactiva el envío al máster del bus.

Esto evita tener que configurar manualmente el ruteo de 12 canales dentro de REAPER.

---

## 22. Configuración multicanal

El modelo Hybrid Demucs utilizado en este proyecto espera siempre una entrada de doce canales:

```text
[1, 12, N]
```

Sin embargo, el complemento permite trabajar con menos canales reales.

Por ejemplo:

```text
2 canales reales  -> 10 canales restantes se rellenan con ceros
4 canales reales  -> 8 canales restantes se rellenan con ceros
12 canales reales -> no se añade relleno
```

La salida mantiene la correspondencia con los canales procesados y REAPER puede redirigir cada señal a una pista independiente mediante el ReaScript.

---

## 23. Procesamiento temporal

REAPER entrega el audio al complemento mediante llamadas sucesivas a `processBlock()`. Cada llamada contiene un bloque de audio con un número determinado de muestras.

El modelo no procesa directamente cada bloque aislado. En su lugar, el complemento mantiene una ventana histórica interna de 8192 muestras.

En cada llamada:

1. Se desplaza el historial anterior.
2. Se incorporan las nuevas muestras recibidas.
3. Se rellenan con ceros los canales no utilizados si hay menos de 12 canales reales.
4. Se construye un tensor de entrada `[1, 12, 8192]`.
5. Se ejecuta el modelo mediante LibTorch.
6. Se reconstruye la salida correspondiente al bloque actual.
7. Se aplica solapamiento-suma para suavizar la unión entre bloques.
8. Se devuelve el audio procesado a REAPER.

---

## 24. Solapamiento-suma

Para evitar discontinuidades entre fragmentos consecutivos, el complemento utiliza un procedimiento de solapamiento-suma.

La configuración final es:

```text
Overlap: 512 muestras
```

El sistema conserva una cola de salida del bloque anterior y la combina con el inicio suavizado del bloque actual. Esta operación permite mejorar la continuidad entre las señales procesadas.

---

## 25. Rendimiento observado

En la configuración principal evaluada:

```text
Sample rate: 48 kHz
Block size: 4096 muestras
Ventana: 8192 muestras
Overlap: 512 muestras
Dispositivo: MPS / Apple Metal
Canales internos: 12
```

se observaron tiempos de inferencia aproximados de:

```text
58-61 ms
```

El tiempo disponible por bloque con 4096 muestras a 48 kHz es aproximadamente:

```text
85,3 ms
```

Por tanto, en el equipo de pruebas la ejecución mediante MPS se mantuvo por debajo del tiempo disponible por bloque.

También se realizaron pruebas con una ventana de 16384 muestras. Aunque esta configuración aportaba más contexto temporal, el tiempo de inferencia superaba el margen disponible y podía generar parones durante la reproducción. Por este motivo, se adoptó la ventana de 8192 muestras como configuración final.

---

## 26. Interfaz gráfica

El plugin incluye una interfaz gráfica informativa.

La interfaz no incorpora parámetros modificables por el usuario. Su finalidad es mostrar de forma clara:

- El objetivo del proyecto.
- Las tecnologías utilizadas.
- La configuración validada.
- El número de canales internos.
- La ventana de contexto.
- El solapamiento.
- El flujo recomendado con REAPER y ReaScript.

El procesamiento del audio no depende de que la ventana del plugin esté abierta.

---

## 27. Solución VST3 desarrollada

La solución VST3 desarrollada sustituye el flujo externo utilizado originalmente por una integración directa dentro de REAPER.

La diferencia puede resumirse así:

```text
Sistema anterior:
REAPER -> BlackHole -> Python -> PyTorch -> BlackHole -> REAPER

Sistema desarrollado:
REAPER -> VST3 Demucs12 -> LibTorch/TorchScript -> REAPER
```

En el sistema anterior, el modelo se ejecutaba fuera de REAPER mediante una aplicación en Python. En cambio, en este TFG el modelo se carga directamente dentro de un complemento VST3 mediante LibTorch.

Esto permite:

- Reducir la dependencia de aplicaciones externas.
- Mantener el procesamiento dentro de la DAW.
- Usar el complemento como un efecto insertable.
- Gestionar el audio mediante el flujo normal de REAPER.
- Automatizar el enrutamiento multicanal mediante ReaScript.
- Aproximar el modelo a un uso más práctico en producción musical.

---

## 28. Problemas frecuentes

### REAPER no detecta el plugin

Comprobar que el archivo `.vst3` está en:

```text
~/Library/Audio/Plug-Ins/VST3/
```

Después, en REAPER:

```text
Preferences -> Plug-ins -> VST -> Re-scan
```

Si sigue sin aparecer, limpiar la caché de plugins de REAPER y volver a escanear.

---

### El plugin carga, pero no encuentra el modelo

Revisar la ruta definida en `loadModel()` dentro de:

```text
PluginProcessor.cpp
```

El archivo debe existir en:

```text
Mabejar99_FinalProjectTFM/exported_models/
```

y debe coincidir con el nombre usado por el código:

```text
demucs_export_torchscript_MPS.pt
```

o:

```text
demucs_export_torchscript_CPU.pt
```

---

### Error con bibliotecas de LibTorch

Comprobar:

- Header Search Paths.
- Library Search Paths.
- External Libraries to Link.
- Extra Linker Flags.
- Ruta `rpath`.
- Que macOS no haya bloqueado las bibliotecas dinámicas.

Puede ser necesario ejecutar:

```bash
xattr -cr /ruta/a/libtorch
```

---

### El script no encuentra el plugin

Abrir el archivo:

```text
scripts/Demucs12_Bus.lua
```

y revisar la variable:

```lua
local pluginName = "VST3: Demucs12 (yourcompany)"
```

El nombre debe coincidir con el que aparece en REAPER. Puede ser necesario probar alguna variante, por ejemplo:

```lua
local pluginName = "Demucs12"
```

o:

```lua
local pluginName = "VST3: Demucs12"
```

---

### No se escuchan las salidas separadas

Comprobar:

1. Que el ReaScript ha creado las pistas separadas.
2. Que el bus tiene el plugin insertado.
3. Que el bus tiene el número correcto de canales.
4. Que las pistas separadas reciben señal desde el bus.
5. Que las pistas separadas tienen activado el envío al máster.
6. Que el bus y las pistas originales tienen desactivado el envío directo al máster.

---

### Hay cortes durante la reproducción

Comprobar:

- Que se está usando MPS y no CPU.
- Que el tamaño de ventana es 8192 muestras.
- Que el bloque de REAPER es 4096 muestras.
- Que el sample rate es 48 kHz.
- Que no se está utilizando una ventana mayor, como 16384 muestras.
- Que el equipo no tiene una carga excesiva durante la prueba.

---

## 29. Limitaciones

Este repositorio contiene un prototipo experimental desarrollado en el contexto de un Trabajo Fin de Grado.

Limitaciones principales:

- El modelo no se entrena en este proyecto.
- La calidad de separación depende del modelo heredado.
- El rendimiento temporal depende del equipo utilizado.
- La versión final se ha validado principalmente en macOS con Apple Silicon.
- La aceleración se realiza mediante MPS, no mediante CUDA.
- El sistema no se ha probado exhaustivamente en otras DAW.
- El complemento no debe considerarse una herramienta comercial terminada.
- El tamaño de ventana introduce una latencia asociada al procesamiento.
- La compatibilidad con otros sistemas requiere adaptar rutas, dependencias y backend de inferencia.

---

## 30. Resultado del proyecto

El resultado final es un prototipo funcional que integra un modelo Hybrid Demucs multicanal dentro de REAPER mediante un complemento VST3.

La solución permite pasar de un modelo ejecutado externamente en Python a un flujo integrado dentro de una DAW:

```text
Antes:
REAPER -> BlackHole -> Python -> PyTorch -> BlackHole -> REAPER

Ahora:
REAPER -> VST3 Demucs12 -> LibTorch/TorchScript -> REAPER
```

Además, el uso de ReaScript permite automatizar la preparación del bus multicanal y de las pistas de salida, acercando el sistema a un flujo de trabajo más cómodo para producción musical.

---

## 31. Referencia del proyecto

Repositorio:

```text
https://github.com/esm00046/TFG-Hybrid-Demucs-VST3
```

Referencia en formato APA:

```text
Martínez, E. S. (2026). TFG-Hybrid-Demucs-VST3 [Código fuente]. GitHub. https://github.com/esm00046/TFG-Hybrid-Demucs-VST3
```

---

## 32. Autor

**Enrique Santiago Martínez**  
Grado en Ingeniería de Tecnologías de Telecomunicación  
Escuela Politécnica Superior de Linares  
Universidad de Jaén

---

## 33. Directores

**Pablo Cabañas Molero**  
Departamento de Ingeniería de Telecomunicación  

**Pedro Vera Candeas**  
Departamento de Ingeniería de Telecomunicación  

---

## 34. Licencia y uso académico

Este repositorio se publica como material asociado a un Trabajo Fin de Grado. Su finalidad principal es documentar el desarrollo realizado, facilitar la revisión del código y permitir la reproducción del prototipo en un entorno compatible.

El uso del modelo heredado, checkpoints, audios y archivos asociados debe respetar las condiciones establecidas por sus autores originales y por la Universidad de Jaén.

---

## 35. Aviso sobre rutas locales

Algunas rutas utilizadas durante el desarrollo corresponden al equipo local en el que se realizó el TFG. Si el proyecto se clona en otro equipo, es necesario revisar y adaptar:

- La ruta de LibTorch.
- La ruta del modelo TorchScript en `PluginProcessor.cpp`.
- La ubicación del archivo `.vst3`.
- El nombre con el que REAPER detecta el plugin.
- La ubicación de los modelos exportados.
- La ubicación del checkpoint heredado.

Sin estos ajustes, el proyecto puede compilar correctamente pero no encontrar el modelo durante la ejecución.

---

## 36. Uso previsto

Este proyecto está pensado como prototipo académico y experimental. Su objetivo es demostrar la viabilidad de integrar un modelo Hybrid Demucs multicanal en un complemento VST3 y ejecutarlo dentro de REAPER con un flujo de audio continuado.

No debe interpretarse como una herramienta comercial final, sino como una base funcional para futuras mejoras, entre ellas:

- Optimización del rendimiento.
- Reducción de la latencia.
- Evaluación objetiva de la calidad de separación.
- Compatibilidad con otros sistemas operativos.
- Compatibilidad con otros formatos de plugin.
- Mejora del empaquetado e instalación.
- Carga dinámica de rutas y modelos.
- Validación en más configuraciones de audio.
