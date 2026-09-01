/*
  ==============================================================================

    This file contains the basic startup code for a JUCE application.

  ==============================================================================
*/
#include <JuceHeader.h>
#include <torch/script.h>

#include <iostream>
#include <chrono>
#include <vector>
#include <algorithm>
#include <cmath>

//==============================================================================
int main (int argc, char* argv[])
{
    /*
    std::cout << "JUCE v" << JUCE_MAJOR_VERSION << "."
    << JUCE_MINOR_VERSION << "."
    << JUCE_BUILDNUMBER << std::endl;
    
    std::cout << "Probando carga de modelo Demucs TorchScript en C++..." << std::endl;
    
    // ============================================================
    // 1. Ruta del modelo TorchScript CPU
    // ============================================================
    //
    // De momento usamos ruta absoluta para evitar errores.
    // Más adelante podemos cambiarlo por una ruta relativa más limpia.
    //
    
    const std::string modelPath =
    "/Users/enriique/Desktop/TFG/TFG_PluginVST3_demucs12/Mabejar99_FinalProjectTFM/exported_models/demucs_export_torchscript_CPU.pt";
    
    try
    {
        // ========================================================
        // 2. Cargar modelo
        // ========================================================
        
        std::cout << "Cargando modelo desde:" << std::endl;
        std::cout << modelPath << std::endl;
        
        torch::jit::script::Module model = torch::jit::load(modelPath);
        
        model.eval();
        
        std::cout << "Modelo cargado correctamente." << std::endl;
        
        // ========================================================
        // 3. Crear entrada falsa
        // ========================================================
        //
        // Igual que en Python:
        //
        //     [1, 12, 16384]
        //
        // batch = 1
        // canales = 12
        // muestras = 16384
        //
        
        const int batchSize = 1;
        const int numChannels = 12;
        const int numSamples = 16384;
        
        torch::Tensor inputTensor = torch::randn(
                                                 {batchSize, numChannels, numSamples},
                                                 torch::TensorOptions().dtype(torch::kFloat32)
                                                 );
        
        std::cout << "Tensor de entrada creado." << std::endl;
        std::cout << "Forma entrada: " << inputTensor.sizes() << std::endl;
        std::cout << "Tipo entrada: float32" << std::endl;
        std::cout << "Dispositivo entrada: CPU" << std::endl;
        
        // ========================================================
        // 4. Pasada de calentamiento
        // ========================================================
        //
        // La primera pasada puede tardar algo más.
        //
        
        std::cout << "Ejecutando pasada de calentamiento..." << std::endl;
        
        {
            torch::NoGradGuard noGrad;
            
            std::vector<torch::jit::IValue> inputs;
            inputs.push_back(inputTensor);
            
            torch::Tensor warmupOutput = model.forward(inputs).toTensor();
            
            std::cout << "Calentamiento completado." << std::endl;
            std::cout << "Forma salida calentamiento: "
            << warmupOutput.sizes() << std::endl;
        }
        
        // ========================================================
        // 5. Medir inferencia
        // ========================================================
        
        std::cout << "Midiendo tiempo de inferencia..." << std::endl;
        
        torch::Tensor outputTensor;
        
        auto start = std::chrono::high_resolution_clock::now();
        
        {
            torch::NoGradGuard noGrad;
            
            std::vector<torch::jit::IValue> inputs;
            inputs.push_back(inputTensor);
            
            outputTensor = model.forward(inputs).toTensor();
        }
        
        auto end = std::chrono::high_resolution_clock::now();
        
        const double elapsedMs =
        std::chrono::duration<double, std::milli>(end - start).count();
        
        // ========================================================
        // 6. Mostrar resultados
        // ========================================================
        
        outputTensor = outputTensor.contiguous();
        
        std::cout << std::endl;
        std::cout << "Resultados:" << std::endl;
        std::cout << "Forma salida: " << outputTensor.sizes() << std::endl;
        std::cout << "Tiempo inferencia: " << elapsedMs << " ms" << std::endl;
        
        std::cout << "Output min: "
        << outputTensor.min().item<float>() << std::endl;
        
        std::cout << "Output max: "
        << outputTensor.max().item<float>() << std::endl;
        
        std::cout << "Output abs max: "
        << outputTensor.abs().max().item<float>() << std::endl;
        
        // ========================================================
        // 7. Comprobación de dimensiones
        // ========================================================
        
        if (outputTensor.dim() == 3)
        {
            const int64_t outBatch = outputTensor.size(0);
            const int64_t outChannels = outputTensor.size(1);
            const int64_t outSamples = outputTensor.size(2);
            
            std::cout << std::endl;
            std::cout << "Interpretacion salida:" << std::endl;
            std::cout << "Batch: " << outBatch << std::endl;
            std::cout << "Canales/salidas: " << outChannels << std::endl;
            std::cout << "Muestras: " << outSamples << std::endl;
            
            if (outChannels == 12)
                std::cout << "Correcto: el modelo devuelve 12 salidas." << std::endl;
            else
                std::cout << "AVISO: el modelo no devuelve 12 salidas." << std::endl;
            
            if (outSamples == numSamples)
                std::cout << "Correcto: la salida tiene el mismo numero de muestras que la entrada." << std::endl;
            else
                std::cout << "AVISO: la salida no tiene el mismo numero de muestras que la entrada." << std::endl;
        }
        else
        {
            std::cout << "AVISO: la salida no tiene 3 dimensiones." << std::endl;
        }
    }
    catch (const c10::Error& e)
    {
        std::cerr << "Error de LibTorch:" << std::endl;
        std::cerr << e.what() << std::endl;
        return 1;
    }
    catch (const std::exception& e)
    {
        std::cerr << "Error estandar:" << std::endl;
        std::cerr << e.what() << std::endl;
        return 1;
    }
    
    std::cout << std::endl;
    std::cout << "Prueba C++ offline finalizada correctamente." << std::endl;
    
    return 0;
    
    */
    /*
    std::cout << "JUCE v" << JUCE_MAJOR_VERSION << "."
                  << JUCE_MINOR_VERSION << "."
                  << JUCE_BUILDNUMBER << std::endl;

        std::cout << "Probando carga de modelo Demucs TorchScript en C++ usando MPS..." << std::endl;

        // ============================================================
        // 1. Ruta del modelo TorchScript MPS
        // ============================================================
        //
        // De momento usamos ruta absoluta para evitar problemas.
        //

        const std::string modelPath =
            "/Users/enriique/Desktop/TFG/TFG_PluginVST3_demucs12/Mabejar99_FinalProjectTFM/exported_models/demucs_export_torchscript_MPS.pt";

        try
        {
            // ========================================================
            // 2. Seleccionar dispositivo MPS
            // ========================================================
            //
            // En Python comprobábamos torch.backends.mps.is_available().
            // En C++ vamos a intentar crear directamente el dispositivo MPS.
            //
            // Si LibTorch no tiene soporte MPS en tu instalación, aquí o en
            // el movimiento del modelo/tensor aparecerá el error.
            //

            torch::Device device(torch::kMPS);

            std::cout << "Dispositivo seleccionado: MPS" << std::endl;

            // ========================================================
            // 3. Cargar modelo
            // ========================================================

            std::cout << "Cargando modelo desde:" << std::endl;
            std::cout << modelPath << std::endl;

            torch::jit::script::Module model = torch::jit::load(modelPath);

            model.eval();

            // Movemos el modelo a MPS.
            model.to(device);

            std::cout << "Modelo cargado correctamente y movido a MPS." << std::endl;

            // ========================================================
            // 4. Crear entrada falsa en MPS
            // ========================================================
            //
            // Igual que en Python:
            //
            //     [1, 12, 16384]
            //
            // batch = 1
            // canales = 12
            // muestras = 16384
            //

            const int batchSize = 1;
            const int numChannels = 12;
            const int numSamples = 16384;

            torch::Tensor inputTensor = torch::randn(
                {batchSize, numChannels, numSamples},
                torch::TensorOptions()
                    .dtype(torch::kFloat32)
                    .device(device)
            );

            std::cout << "Tensor de entrada creado." << std::endl;
            std::cout << "Forma entrada: " << inputTensor.sizes() << std::endl;
            std::cout << "Dispositivo entrada: " << inputTensor.device() << std::endl;
            std::cout << "Tipo entrada: float32" << std::endl;

            // ========================================================
            // 5. Pasada de calentamiento
            // ========================================================
            //
            // La primera pasada suele tardar más.
            // Para asegurar que MPS ha terminado, movemos la salida a CPU.
            //

            std::cout << "Ejecutando pasada de calentamiento..." << std::endl;

            {
                torch::NoGradGuard noGrad;

                std::vector<torch::jit::IValue> inputs;
                inputs.push_back(inputTensor);

                torch::Tensor warmupOutputMPS = model.forward(inputs).toTensor();

                std::cout << "Forma salida calentamiento MPS: "
                          << warmupOutputMPS.sizes() << std::endl;

                std::cout << "Dispositivo salida calentamiento: "
                          << warmupOutputMPS.device() << std::endl;

                torch::Tensor warmupOutputCPU =
                    warmupOutputMPS.to(torch::kCPU).contiguous();

                std::cout << "Calentamiento completado." << std::endl;
                std::cout << "Forma salida calentamiento CPU: "
                          << warmupOutputCPU.sizes() << std::endl;
            }

            // ========================================================
            // 6. Medir inferencia
            // ========================================================
            //
            // Incluimos en la medida el paso a CPU para forzar que MPS
            // haya terminado realmente el cálculo.
            //

            std::cout << "Midiendo tiempo de inferencia..." << std::endl;

            torch::Tensor outputTensorMPS;
            torch::Tensor outputTensorCPU;

            auto start = std::chrono::high_resolution_clock::now();

            {
                torch::NoGradGuard noGrad;

                std::vector<torch::jit::IValue> inputs;
                inputs.push_back(inputTensor);

                outputTensorMPS = model.forward(inputs).toTensor();

                // Forzamos sincronización llevando la salida a CPU.
                outputTensorCPU = outputTensorMPS.to(torch::kCPU).contiguous();
            }

            auto end = std::chrono::high_resolution_clock::now();

            const double elapsedMs =
                std::chrono::duration<double, std::milli>(end - start).count();

            // ========================================================
            // 7. Mostrar resultados
            // ========================================================

            std::cout << std::endl;
            std::cout << "Resultados:" << std::endl;

            std::cout << "Forma salida MPS: " << outputTensorMPS.sizes() << std::endl;
            std::cout << "Dispositivo salida MPS: " << outputTensorMPS.device() << std::endl;

            std::cout << "Forma salida CPU: " << outputTensorCPU.sizes() << std::endl;
            std::cout << "Dispositivo salida CPU: " << outputTensorCPU.device() << std::endl;

            std::cout << "Tiempo inferencia MPS + copia CPU: "
                      << elapsedMs << " ms" << std::endl;

            std::cout << "Output min: "
                      << outputTensorCPU.min().item<float>() << std::endl;

            std::cout << "Output max: "
                      << outputTensorCPU.max().item<float>() << std::endl;

            std::cout << "Output abs max: "
                      << outputTensorCPU.abs().max().item<float>() << std::endl;

            // ========================================================
            // 8. Comprobación de dimensiones
            // ========================================================

            if (outputTensorCPU.dim() == 3)
            {
                const int64_t outBatch = outputTensorCPU.size(0);
                const int64_t outChannels = outputTensorCPU.size(1);
                const int64_t outSamples = outputTensorCPU.size(2);

                std::cout << std::endl;
                std::cout << "Interpretacion salida:" << std::endl;
                std::cout << "Batch: " << outBatch << std::endl;
                std::cout << "Canales/salidas: " << outChannels << std::endl;
                std::cout << "Muestras: " << outSamples << std::endl;

                if (outChannels == 12)
                    std::cout << "Correcto: el modelo devuelve 12 salidas." << std::endl;
                else
                    std::cout << "AVISO: el modelo no devuelve 12 salidas." << std::endl;

                if (outSamples == numSamples)
                    std::cout << "Correcto: la salida tiene el mismo numero de muestras que la entrada." << std::endl;
                else
                    std::cout << "AVISO: la salida no tiene el mismo numero de muestras que la entrada." << std::endl;
            }
            else
            {
                std::cout << "AVISO: la salida no tiene 3 dimensiones." << std::endl;
            }
        }
        catch (const c10::Error& e)
        {
            std::cerr << "Error de LibTorch:" << std::endl;
            std::cerr << e.what() << std::endl;
            return 1;
        }
        catch (const std::exception& e)
        {
            std::cerr << "Error estandar:" << std::endl;
            std::cerr << e.what() << std::endl;
            return 1;
        }

        std::cout << std::endl;
        std::cout << "Prueba C++ offline MPS finalizada correctamente." << std::endl;

        return 0;
    */
    
    
    std::cout << "JUCE v" << JUCE_MAJOR_VERSION << "."
              << JUCE_MINOR_VERSION << "."
              << JUCE_BUILDNUMBER << std::endl;

    std::cout << "Prueba offline Demucs 12 canales con LibTorch CPU" << std::endl;

    // ============================================================
    // 1. Rutas
    // ============================================================

    const std::string modelPath =
        "/Users/enriique/Desktop/TFG/TFG_PluginVST3_demucs12/Mabejar99_FinalProjectTFM/exported_models/demucs_export_torchscript_CPU.pt";

    const juce::File inputFile(
        "/Users/enriique/Desktop/TFG/TFG_PluginVST3_demucs12/Mabejar99_FinalProjectTFM/audio/Freischutz/6/mixture.wav"
    );

    const juce::File outputFolder(
        "/Users/enriique/Desktop/TFG/TFG_PluginVST3_demucs12/cpp_offline_tests/output_audio"
    );

    outputFolder.createDirectory();

    // ============================================================
    // 2. Parámetros del modelo
    // ============================================================

    const int modelChannels = 12;
    const int modelSamples = 4096;
    const double expectedSampleRate = 48000.0;

    try
    {
        // ========================================================
        // 3. Cargar modelo TorchScript CPU
        // ========================================================

        std::cout << "\nCargando modelo:" << std::endl;
        std::cout << modelPath << std::endl;

        torch::jit::script::Module model = torch::jit::load(modelPath);
        model.eval();

        std::cout << "Modelo cargado correctamente." << std::endl;

        // ========================================================
        // 4. Leer archivo WAV
        // ========================================================

        if (!inputFile.existsAsFile())
        {
            std::cerr << "No se ha encontrado el archivo de entrada:" << std::endl;
            std::cerr << inputFile.getFullPathName() << std::endl;
            return 1;
        }

        juce::AudioFormatManager formatManager;
        formatManager.registerBasicFormats();

        std::unique_ptr<juce::AudioFormatReader> reader(
            formatManager.createReaderFor(inputFile)
        );

        if (reader == nullptr)
        {
            std::cerr << "No se ha podido abrir el WAV de entrada." << std::endl;
            return 1;
        }

        const double sampleRate = reader->sampleRate;
        const int inputChannels = static_cast<int>(reader->numChannels);
        const int totalSamples = static_cast<int>(reader->lengthInSamples);

        std::cout << "\nAudio de entrada:" << std::endl;
        std::cout << "Ruta: " << inputFile.getFullPathName() << std::endl;
        std::cout << "Sample rate: " << sampleRate << " Hz" << std::endl;
        std::cout << "Canales del WAV: " << inputChannels << std::endl;
        std::cout << "Muestras totales: " << totalSamples << std::endl;

        if (std::abs(sampleRate - expectedSampleRate) > 1.0)
        {
            std::cerr << "\nERROR: El modelo está preparado para 48000 Hz." << std::endl;
            std::cerr << "El WAV tiene sample rate: " << sampleRate << " Hz" << std::endl;
            std::cerr << "De momento no hacemos resampleo en esta prueba offline." << std::endl;
            return 1;
        }

        if (totalSamples <= 0)
        {
            std::cerr << "El archivo no contiene muestras válidas." << std::endl;
            return 1;
        }

        // Usamos hasta 12 canales.
        // Si el WAV tiene menos canales, los restantes se rellenarán con ceros.
        const int channelsToRead = std::min(inputChannels, modelChannels);

        juce::AudioBuffer<float> inputBuffer(channelsToRead, totalSamples);
        inputBuffer.clear();

        // ========================================================
        // Lectura sencilla del WAV directamente a AudioBuffer<float>
        // ========================================================
        //
        // Esta llamada evita trabajar manualmente con punteros int**.
        // Para esta primera validación offline nos vale porque queremos
        // comprobar el flujo completo:
        //
        // WAV -> tensor [1, 12, 16384] -> modelo -> 12 salidas WAV
        //

        const bool okRead = reader->read(
            &inputBuffer,       // buffer destino
            0,                  // primera muestra donde escribir
            totalSamples,       // número de muestras a leer
            0,                  // primera muestra del archivo
            true,               // leer canal izquierdo / canal 0
            true                // leer canal derecho / canal 1
        );

        if (!okRead)
        {
            std::cerr << "Error leyendo las muestras del WAV." << std::endl;
            return 1;
        }

        if (inputChannels < modelChannels)
        {
            std::cout << "\nAVISO: el WAV tiene menos de 12 canales." << std::endl;
            std::cout << "Los canales restantes se rellenarán con ceros." << std::endl;
        }

        if (inputChannels > modelChannels)
        {
            std::cout << "\nAVISO: el WAV tiene más de 12 canales." << std::endl;
            std::cout << "Solo se usarán los primeros 12 canales." << std::endl;
        }

        // ========================================================
        // 5. Preparar almacenamiento de las 12 salidas
        // ========================================================

        std::vector<std::vector<float>> outputChannels(
            modelChannels,
            std::vector<float>(totalSamples, 0.0f)
        );

        // Buffer temporal de entrada al modelo.
        //
        // Forma lógica:
        //     [1, 12, 16384]
        //
        // En memoria:
        //     canal 0 completo,
        //     canal 1 completo,
        //     ...
        //     canal 11 completo.
        //
        std::vector<float> inputChunk(modelChannels * modelSamples, 0.0f);

        const int numChunks = (totalSamples + modelSamples - 1) / modelSamples;

        std::cout << "\nProcesando audio por bloques:" << std::endl;
        std::cout << "Tamaño de bloque del modelo: " << modelSamples << " muestras" << std::endl;
        std::cout << "Número total de bloques: " << numChunks << std::endl;

        auto totalStart = std::chrono::high_resolution_clock::now();

        // ========================================================
        // 6. Procesar por bloques
        // ========================================================

        for (int chunkIndex = 0; chunkIndex < numChunks; ++chunkIndex)
        {
            const int startSample = chunkIndex * modelSamples;
            const int remainingSamples = totalSamples - startSample;
            const int currentSamples = std::min(modelSamples, remainingSamples);

            std::fill(inputChunk.begin(), inputChunk.end(), 0.0f);

            // Copiar audio real al chunk.
            // Si faltan canales, los que no existan se quedan a cero.
            for (int c = 0; c < modelChannels; ++c)
            {
                float* chunkChannelData = inputChunk.data() + c * modelSamples;

                if (c < channelsToRead)
                {
                    const float* inputData = inputBuffer.getReadPointer(c);

                    std::copy(inputData + startSample,
                              inputData + startSample + currentSamples,
                              chunkChannelData);
                }
            }

            torch::Tensor inputTensor = torch::from_blob(
                inputChunk.data(),
                {1, modelChannels, modelSamples},
                torch::TensorOptions().dtype(torch::kFloat32)
            ).clone();

            torch::Tensor outputTensor;

            auto chunkStart = std::chrono::high_resolution_clock::now();

            {
                torch::NoGradGuard noGrad;

                std::vector<torch::jit::IValue> inputs;
                inputs.push_back(inputTensor);

                outputTensor = model.forward(inputs).toTensor();
            }

            auto chunkEnd = std::chrono::high_resolution_clock::now();

            const double chunkMs =
                std::chrono::duration<double, std::milli>(chunkEnd - chunkStart).count();

            outputTensor = outputTensor.to(torch::kCPU).contiguous();

            if (outputTensor.dim() != 3)
            {
                std::cerr << "ERROR: la salida del modelo no tiene 3 dimensiones." << std::endl;
                return 1;
            }

            const int64_t outBatch = outputTensor.size(0);
            const int64_t outChannels = outputTensor.size(1);
            const int64_t outSamples = outputTensor.size(2);

            if (outBatch != 1 || outChannels != modelChannels || outSamples < currentSamples)
            {
                std::cerr << "ERROR: forma de salida inesperada." << std::endl;
                std::cerr << "Salida: " << outputTensor.sizes() << std::endl;
                return 1;
            }

            const float* outPtr = outputTensor.data_ptr<float>();

            // Copiar cada una de las 12 salidas al buffer completo.
            for (int c = 0; c < modelChannels; ++c)
            {
                const float* sourceData = outPtr + c * outSamples;

                std::copy(sourceData,
                          sourceData + currentSamples,
                          outputChannels[c].begin() + startSample);
            }

            std::cout << "Bloque " << (chunkIndex + 1) << " / " << numChunks
                      << " procesado en " << chunkMs << " ms" << std::endl;
        }

        auto totalEnd = std::chrono::high_resolution_clock::now();

        const double totalMs =
            std::chrono::duration<double, std::milli>(totalEnd - totalStart).count();

        std::cout << "\nProcesamiento completado." << std::endl;
        std::cout << "Tiempo total: " << totalMs << " ms" << std::endl;

        // ========================================================
        // 7. Normalizar salida como en el notebook
        // ========================================================
        //
        // En el cuaderno se hacía:
        //
        //     separated = separated / separated.abs().max()
        //
        // Aquí hacemos algo equivalente para evitar clipping al guardar.
        //

        float maxAbs = 0.0f;

        for (int c = 0; c < modelChannels; ++c)
        {
            for (float v : outputChannels[c])
                maxAbs = std::max(maxAbs, std::abs(v));
        }

        std::cout << "Máximo absoluto antes de normalizar: " << maxAbs << std::endl;

        if (maxAbs > 1.0e-8f)
        {
            for (int c = 0; c < modelChannels; ++c)
            {
                for (float& v : outputChannels[c])
                    v /= maxAbs;
            }

            std::cout << "Salidas normalizadas." << std::endl;
        }

        // ========================================================
        // 8. Guardar las 12 salidas como WAV mono
        // ========================================================

        juce::WavAudioFormat wavFormat;
        juce::StringPairArray metadata;

        std::cout << "\nGuardando salidas en:" << std::endl;
        std::cout << outputFolder.getFullPathName() << std::endl;

        for (int c = 0; c < modelChannels; ++c)
        {
            juce::File outFile =
                outputFolder.getChildFile("source_" + juce::String(c) + ".wav");

            if (outFile.existsAsFile())
                outFile.deleteFile();

            std::unique_ptr<juce::FileOutputStream> stream(
                outFile.createOutputStream()
            );

            if (stream == nullptr)
            {
                std::cerr << "No se pudo crear el archivo: "
                          << outFile.getFullPathName() << std::endl;
                return 1;
            }

            std::unique_ptr<juce::AudioFormatWriter> writer(
                wavFormat.createWriterFor(
                    stream.get(),
                    sampleRate,
                    1,
                    24,
                    metadata,
                    0
                )
            );

            if (writer == nullptr)
            {
                std::cerr << "No se pudo crear el writer WAV para: "
                          << outFile.getFullPathName() << std::endl;
                return 1;
            }

            stream.release();

            juce::AudioBuffer<float> outputBuffer(1, totalSamples);
            outputBuffer.clear();

            outputBuffer.copyFrom(
                0,
                0,
                outputChannels[c].data(),
                totalSamples
            );

            const bool okWrite = writer->writeFromAudioSampleBuffer(
                outputBuffer,
                0,
                totalSamples
            );

            if (!okWrite)
            {
                std::cerr << "Error guardando: "
                          << outFile.getFullPathName() << std::endl;
                return 1;
            }

            std::cout << "Guardado: " << outFile.getFileName() << std::endl;
        }

        std::cout << "\nPrueba offline finalizada correctamente." << std::endl;
    }
    catch (const c10::Error& e)
    {
        std::cerr << "Error de LibTorch:" << std::endl;
        std::cerr << e.what() << std::endl;
        return 1;
    }
    catch (const std::exception& e)
    {
        std::cerr << "Error estándar:" << std::endl;
        std::cerr << e.what() << std::endl;
        return 1;
    }

    return 0;
    
}
