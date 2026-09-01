
#include "PluginProcessor.h"
#include "PluginEditor.h"

#include <iostream>
#include <string>
#include <vector>
#include <algorithm>
#include <cstring>
#include <cmath>
//==============================================================================
//Constructor
//==============================================================================
Demucs12AudioProcessor::Demucs12AudioProcessor()
    : AudioProcessor(BusesProperties()
        .withInput("Input", juce::AudioChannelSet::stereo(), true)
        .withOutput("Output", juce::AudioChannelSet::stereo(), true))
{
    isModelLoaded = loadModel();
}
/*
Demucs12AudioProcessor::Demucs12AudioProcessor()
: AudioProcessor(BusesProperties()
                 .withInput("Input", juce::AudioChannelSet::discreteChannels(modelChannels), true)
                 .withOutput("Output", juce::AudioChannelSet::discreteChannels(modelChannels), true))
{
    isModelLoaded = loadModel();
}
*/

Demucs12AudioProcessor::~Demucs12AudioProcessor()
{
}

//==============================================================================


//==============================================================================
//Loadmodel
//==============================================================================
bool Demucs12AudioProcessor::loadModel()
{
    std::string modelpath =
    "/Users/enriique/Desktop/TFG/TFG_PluginVST3_demucs12/Mabejar99_FinalProjectTFM/exported_models/";
    
    try
    {
        // ========================================================
        // 1. Selección del dispositivo
        // ========================================================
        //
        // Primero intentamos usar MPS/Metal, que es la opción GPU en Mac.
        // Si no estuviera disponible, usamos CPU como segunda opción.
        //
        
        if (torch::mps::is_available())
        {
            modelDevice = torch::Device(torch::kMPS);
            
            std::cout << "MPS is available. Using Apple Metal GPU." << std::endl;
            
            modelpath += "demucs_export_torchscript_MPS.pt";
        }
        else
        {
            modelDevice = torch::Device(torch::kCPU);
            
            std::cout << "MPS not available. Using CPU." << std::endl;
            
            modelpath += "demucs_export_torchscript_CPU.pt";
        }
        
        // ========================================================
        // 2. Carga del modelo TorchScript
        // ========================================================
        
        std::cout << "Loading model from:" << std::endl;
        std::cout << modelpath << std::endl;
        
        modelDemucs = torch::jit::load(modelpath);
        
        // Movemos el modelo al dispositivo seleccionado.
        modelDemucs.to(modelDevice);
        
        // Modo inferencia/evaluación.
        modelDemucs.eval();
        
        std::cout << "Model loaded successfully." << std::endl;
        
        return true;
    }
    catch (const c10::Error& e)
    {
        std::cerr << "Error loading the model: " << e.what() << std::endl;
        return false;
    }
}
//==============================================================================
void Demucs12AudioProcessor::createPerfectHann(int overlapSamples)
{
    const int L = overlapSamples;
    const int N = 2 * L;

    const float pi = juce::MathConstants<float>::pi;

    std::vector<float> hann(N, 0.0f);

    for (int n = 0; n < N; ++n)
    {
        hann[n] = 0.5f * (1.0f - std::cos(2.0f * pi * n / N));
    }

    // Primera mitad: fade-in.
    winFirstHalf.assign(hann.begin(),
                        hann.begin() + L);

    // Segunda mitad: fade-out.
    winSecondHalf.assign(hann.begin() + L,
                         hann.end());
}

//==============================================================================
const juce::String Demucs12AudioProcessor::getName() const
{
    return JucePlugin_Name;
}

bool Demucs12AudioProcessor::acceptsMidi() const
{
#if JucePlugin_WantsMidiInput
    return true;
#else
    return false;
#endif
}

bool Demucs12AudioProcessor::producesMidi() const
{
#if JucePlugin_ProducesMidiOutput
    return true;
#else
    return false;
#endif
}

bool Demucs12AudioProcessor::isMidiEffect() const
{
#if JucePlugin_IsMidiEffect
    return true;
#else
    return false;
#endif
}

double Demucs12AudioProcessor::getTailLengthSeconds() const
{
    return 0.0;
}

int Demucs12AudioProcessor::getNumPrograms()
{
    return 1;   // NB: some hosts don't cope very well if you tell them there are 0 programs,
    // so this should be at least 1, even if you're not really implementing programs.
}

int Demucs12AudioProcessor::getCurrentProgram()
{
    return 0;
}

void Demucs12AudioProcessor::setCurrentProgram (int index)
{
}

const juce::String Demucs12AudioProcessor::getProgramName (int index)
{
    return {};
}

void Demucs12AudioProcessor::changeProgramName (int index, const juce::String& newName)
{
}
//==============================================================================


//==============================================================================
//Preparetoplay
//==============================================================================
void Demucs12AudioProcessor::prepareToPlay(double sampleRate, int blockSize)
{
    std::cout << "Current host sample rate: " << sampleRate << " Hz" << std::endl;
    std::cout << "Current host block size: " << blockSize << " samples" << std::endl;

    // ============================================================
    // Comprobación de frecuencia
    // ============================================================
    //
    // Este modelo trabaja a 48 kHz, por lo que no hacemos resampleo.
    //

    if (std::abs(sampleRate - 48000.0) > 1.0)
    {
        std::cout << "WARNING: Demucs12 está preparado para 48000 Hz." << std::endl;
        std::cout << "Sample rate actual del host: " << sampleRate << " Hz" << std::endl;
        std::cout << "En esta versión no se realiza resampleo interno." << std::endl;
    }

    // ============================================================
    // Protección del tamaño de solape
    // ============================================================

    if (overlapSamples < 1)
        overlapSamples = 1;

    if (overlapSamples >= blockSize)
    {
        overlapSamples = std::max(1, blockSize / 4);

        std::cout << "WARNING: overlapSamples era demasiado grande." << std::endl;
        std::cout << "Nuevo overlapSamples: " << overlapSamples << std::endl;
    }

    // ============================================================
    // Inicialización de la ventana histórica del modelo
    // ============================================================
    //
    // modelInput contiene 12 canales consecutivos:
    //
    // canal 0 -> modelInput[0 * modelInputSamples + i]
    // canal 1 -> modelInput[1 * modelInputSamples + i]
    // ...
    // canal 11 -> modelInput[11 * modelInputSamples + i]
    //

    modelInput.assign(modelChannels * modelInputSamples, 0.0f);

    // ============================================================
    // Inicialización de la cola previa para overlap-add
    // ============================================================
    //
    // prevTailOut contiene:
    //
    // canal 0 -> prevTailOut[0 * overlapSamples + i]
    // canal 1 -> prevTailOut[1 * overlapSamples + i]
    // ...
    // canal 11 -> prevTailOut[11 * overlapSamples + i]
    //

    prevTailOut.assign(modelChannels * overlapSamples, 0.0f);

    // Creamos las ventanas Hann usadas para mezclar bloques.
    createPerfectHann(overlapSamples);

    // Al empezar no existe una cola anterior real.
    hasPreviousTail = false;

    std::cout << "Model channels: " << modelChannels << std::endl;
    std::cout << "Model input samples: " << modelInputSamples << std::endl;
    std::cout << "Model input total size: " << modelInput.size() << " samples" << std::endl;
    std::cout << "Overlap samples: " << overlapSamples << std::endl;
}
//==============================================================================
void Demucs12AudioProcessor::releaseResources()
{
}
/*
#ifndef JucePlugin_PreferredChannelConfigurations
bool Demucs12AudioProcessor::isBusesLayoutSupported(const BusesLayout& layouts) const
{
#if JucePlugin_IsMidiEffect
    
    juce::ignoreUnused(layouts);
    return true;
    
#else
    
    const auto& inputLayout = layouts.getMainInputChannelSet();
    const auto& outputLayout = layouts.getMainOutputChannelSet();
    
    if (inputLayout != outputLayout)
        return false;
    
    return inputLayout == juce::AudioChannelSet::discreteChannels(modelChannels);
    
#endif
}
#endif
 */
bool Demucs12AudioProcessor::isBusesLayoutSupported(const BusesLayout& layouts) const
{
    const auto& inputLayout = layouts.getMainInputChannelSet();
    const auto& outputLayout = layouts.getMainOutputChannelSet();

    // La entrada y la salida deben tener el mismo número de canales reales.
    // Ejemplo válido:
    //      2 entradas -> 2 salidas
    //      4 entradas -> 4 salidas
    //      12 entradas -> 12 salidas
    //
    // Ejemplo no válido:
    //      2 entradas -> 12 salidas
    //      12 entradas -> 2 salidas
    //
    if (inputLayout != outputLayout)
        return false;

    const int numberOfChannels = inputLayout.size();

    // El modelo interno siempre trabaja con 12 canales, pero el usuario
    // puede usar el plugin con menos canales reales.
    //
    // Internamente los canales que falten se rellenarán con ceros.
    //
    return numberOfChannels >= 1 && numberOfChannels <= modelChannels;
}

//==============================================================================


//==============================================================================
//ProcessBlock
//==============================================================================
void Demucs12AudioProcessor::processBlock(juce::AudioBuffer<float>& buffer,
                                          juce::MidiBuffer& midiMessages)
{
    juce::ScopedNoDenormals noDenormals;
    juce::ignoreUnused(midiMessages);

    const double inicioInferencia = juce::Time::getMillisecondCounterHiRes();

    // Si el modelo no ha cargado correctamente, dejamos pasar el audio sin modificar.
    if (!isModelLoaded)
        return;

    const int B = buffer.getNumSamples();       // Bloque real que entrega REAPER.
    const int bufferChannels = buffer.getNumChannels();

    const int N = modelInputSamples;            // Ventana completa que entra al modelo.
    const int L = overlapSamples;               // Tamaño del solape.

    if (B <= 0 || bufferChannels <= 0)
        return;

    // ============================================================
    // 1. Número de canales reales de REAPER
    // ============================================================
    //
    // El modelo siempre necesita 12 canales.
    // Sin embargo, REAPER puede estar trabajando con menos canales:
    //
    //      2 canales reales  -> canales 3-12 se rellenan con ceros
    //      4 canales reales  -> canales 5-12 se rellenan con ceros
    //      12 canales reales -> no se rellena nada
    //
    // realInputChannels indica cuántos canales de entrada reales tenemos.
    // realOutputChannels indica cuántos canales vamos a devolver realmente a REAPER.
    //

    const int realInputChannels = std::min({ getTotalNumInputChannels(),
                                             bufferChannels,
                                             modelChannels });

    const int realOutputChannels = std::min({ getTotalNumOutputChannels(),
                                              bufferChannels,
                                              modelChannels });

    // ============================================================
    // 2. Comprobaciones de seguridad
    // ============================================================

    if (static_cast<int>(modelInput.size()) != modelChannels * modelInputSamples)
        modelInput.assign(modelChannels * modelInputSamples, 0.0f);

    if (static_cast<int>(prevTailOut.size()) != modelChannels * overlapSamples)
        prevTailOut.assign(modelChannels * overlapSamples, 0.0f);

    if (B <= L)
    {
        std::cout << "ERROR: block size must be greater than overlap."
                  << " B: " << B
                  << " | L: " << L
                  << std::endl;

        return;
    }

    if (B + L > N)
    {
        std::cout << "ERROR: model window too small for overlap-add."
                  << " B: " << B
                  << " | L: " << L
                  << " | N: " << N
                  << std::endl;

        return;
    }

    // ============================================================
    // 3. Actualizar entrada histórica del modelo
    // ============================================================
    //
    // modelInput tiene siempre 12 canales, aunque REAPER tenga menos.
    //
    // Organización:
    //
    //      canal 0 -> modelInput[0 * N + i]
    //      canal 1 -> modelInput[1 * N + i]
    //      ...
    //      canal 11 -> modelInput[11 * N + i]
    //
    // Para cada canal:
    //      1. Desplazamos el historial B muestras hacia la izquierda.
    //      2. Copiamos al final el bloque nuevo de REAPER.
    //      3. Si ese canal no existe en REAPER, rellenamos con ceros.
    //

    for (int c = 0; c < modelChannels; ++c)
    {
        float* channelInput = modelInput.data() + c * N;

        // Desplazamiento de la ventana histórica.
        std::memmove(channelInput,
                     channelInput + B,
                     (N - B) * sizeof(float));

        if (c < realInputChannels)
        {
            // Canal real procedente de REAPER.
            const float* inputData = buffer.getReadPointer(c);

            std::memcpy(channelInput + (N - B),
                        inputData,
                        B * sizeof(float));
        }
        else
        {
            // Canal artificial relleno con ceros.
            // Esto replica la lógica del cuaderno de Jupyter:
            // la red siempre recibe 12 canales aunque no todos existan realmente.
            std::fill(channelInput + (N - B),
                      channelInput + N,
                      0.0f);
        }
    }

    try
    {
        // ========================================================
        // 4. Crear tensor de entrada [1, 12, N]
        // ========================================================

        torch::Tensor inputTensor = torch::from_blob(
            modelInput.data(),
            {1, modelChannels, N},
            torch::TensorOptions().dtype(torch::kFloat32)
        ).clone();

        inputTensor = inputTensor.to(modelDevice);

        // ========================================================
        // 5. Inferencia del modelo
        // ========================================================

        torch::Tensor outputTensor;

        {
            torch::NoGradGuard noGrad;

            std::vector<torch::jit::IValue> inputs;
            inputs.push_back(inputTensor);

            outputTensor = modelDemucs.forward(inputs).toTensor();
        }

        // La salida vuelve a CPU para poder copiarla al buffer de JUCE.
        outputTensor = outputTensor.to(torch::kCPU).contiguous();

        // ========================================================
        // 6. Comprobar forma de salida
        // ========================================================
        //
        // Esperamos algo del estilo:
        //
        //      [1, 12, outputSamples]
        //

        if (outputTensor.dim() != 3)
        {
            std::cout << "ERROR: model output has wrong number of dimensions."
                      << " Output dimensions: " << outputTensor.dim()
                      << std::endl;

            return;
        }

        const int64_t outBatch = outputTensor.size(0);
        const int64_t outChannels = outputTensor.size(1);
        const int64_t outSamples64 = outputTensor.size(2);

        const int outSamples = static_cast<int>(outSamples64);

        if (outBatch != 1 ||
            outChannels < modelChannels ||
            outSamples < B + L)
        {
            std::cout << "ERROR: unexpected model output shape."
                      << " Output shape: " << outputTensor.sizes()
                      << " | Needed at least: [1, "
                      << modelChannels << ", "
                      << (B + L) << "]"
                      << std::endl;

            return;
        }

        const float* outPtr = outputTensor.data_ptr<float>();

        // ========================================================
        // 7. Posiciones de bloque útil y cola
        // ========================================================
        //
        // Igual que en el denoiser:
        //
        // blockStart:
        //      inicio del bloque útil que se devuelve a REAPER.
        //
        // tailStart:
        //      inicio de la cola que se guarda para mezclarla con
        //      el siguiente bloque.
        //

        const int blockStart = outSamples - (B + L);
        const int tailStart  = outSamples - L;

        if (blockStart < 0 || tailStart < 0)
        {
            std::cout << "ERROR: invalid blockStart/tailStart."
                      << " blockStart: " << blockStart
                      << " | tailStart: " << tailStart
                      << " | outSamples: " << outSamples
                      << " | B: " << B
                      << " | L: " << L
                      << std::endl;

            return;
        }

        // ========================================================
        // 8. Copiar salida SOLO a los canales reales de REAPER
        // ========================================================
        //
        // La red produce 12 salidas, pero si REAPER está trabajando
        // con menos canales, solo devolvemos esos canales reales.
        //
        // Ejemplo:
        //      REAPER tiene 2 canales  -> devolvemos salidas 1 y 2
        //      REAPER tiene 4 canales  -> devolvemos salidas 1, 2, 3 y 4
        //      REAPER tiene 12 canales -> devolvemos las 12 salidas
        //

        for (int c = 0; c < realOutputChannels; ++c)
        {
            float* outputData = buffer.getWritePointer(c);

            const float* modelChannelOut = outPtr + c * outSamples;
            float* previousTail = prevTailOut.data() + c * L;

            // ----------------------------------------------------
            // Primer tramo: solapamiento-suma
            // ----------------------------------------------------
            //
            // Mezclamos:
            //      cola anterior suavizada
            //      +
            //      inicio del bloque actual suavizado
            //

            for (int i = 0; i < L; ++i)
            {
                const float currentFadeIn =
                    modelChannelOut[blockStart + i] * winFirstHalf[i];

                if (hasPreviousTail)
                    outputData[i] = previousTail[i] + currentFadeIn;
                else
                    outputData[i] = modelChannelOut[blockStart + i];
            }

            // ----------------------------------------------------
            // Segundo tramo: parte directa del bloque
            // ----------------------------------------------------

            for (int i = L; i < B; ++i)
            {
                outputData[i] = modelChannelOut[blockStart + i];
            }
        }

        // ========================================================
        // 9. Actualizar cola previa de los 12 canales
        // ========================================================
        //
        // Aunque REAPER esté trabajando con menos canales, actualizamos
        // la cola de los 12 canales internos para mantener coherente
        // el estado del modelo.
        //

        for (int c = 0; c < modelChannels; ++c)
        {
            const float* modelChannelOut = outPtr + c * outSamples;
            float* previousTail = prevTailOut.data() + c * L;

            for (int i = 0; i < L; ++i)
            {
                previousTail[i] =
                    modelChannelOut[tailStart + i] * winSecondHalf[i];
            }
        }

        hasPreviousTail = true;

        // ========================================================
        // 10. Limpiar canales que no correspondan a salidas reales
        // ========================================================
        //
        // Si el buffer de JUCE tuviera canales extra, se limpian para
        // evitar basura, ruido o realimentaciones.
        //

        for (int c = realOutputChannels; c < bufferChannels; ++c)
        {
            buffer.clear(c, 0, B);
        }

        // ========================================================
        // 11. Protección de salida
        // ========================================================

        for (int c = 0; c < realOutputChannels; ++c)
        {
            float* outputData = buffer.getWritePointer(c);

            for (int i = 0; i < B; ++i)
            {
                if (!std::isfinite(outputData[i]))
                    outputData[i] = 0.0f;

                outputData[i] = juce::jlimit(-0.95f, 0.95f, outputData[i]);
            }
        }

        // ========================================================
        // 12. Medición de tiempo
        // ========================================================

        const double finInferencia = juce::Time::getMillisecondCounterHiRes();
        const double tiempoInferenciaMs = finInferencia - inicioInferencia;

        static int debugCounter = 0;
        debugCounter++;

        if (debugCounter % 50 == 0)
        {
            std::cout << "Inference time: "
                      << tiempoInferenciaMs
                      << " ms | Samples: "
                      << B
                      << " | Real input channels: "
                      << realInputChannels
                      << " | Real output channels: "
                      << realOutputChannels
                      << " | Model channels: "
                      << modelChannels
                      << " | Window: "
                      << N
                      << " | Overlap: "
                      << L
                      << " | Device: "
                      << (modelDevice.is_mps() ? "MPS" : "CPU")
                      << std::endl;
        }
    }
    catch (const c10::Error& e)
    {
        std::cerr << "Error during Demucs12 inference: "
                  << e.what()
                  << std::endl;
    }
    catch (const std::exception& e)
    {
        std::cerr << "Standard exception during Demucs12 inference: "
                  << e.what()
                  << std::endl;
    }
}
//==============================================================================
bool Demucs12AudioProcessor::hasEditor() const
{
    return true; // (change this to false if you choose to not supply an editor)
}

juce::AudioProcessorEditor* Demucs12AudioProcessor::createEditor()
{
    return new Demucs12AudioProcessorEditor (*this);
}
//==============================================================================

//==============================================================================
void Demucs12AudioProcessor::getStateInformation (juce::MemoryBlock& destData)
{
    // You should use this method to store your parameters in the memory block.
    // You could do that either as raw data, or use the XML or ValueTree classes
    // as intermediaries to make it easy to save and load complex data.
}

void Demucs12AudioProcessor::setStateInformation (const void* data, int sizeInBytes)
{
    // You should use this method to restore your parameters from this memory block,
    // whose contents will have been created by the getStateInformation() call.
}

//==============================================================================
// This creates new instances of the plugin..
juce::AudioProcessor* JUCE_CALLTYPE createPluginFilter()
{
    return new Demucs12AudioProcessor();
}
