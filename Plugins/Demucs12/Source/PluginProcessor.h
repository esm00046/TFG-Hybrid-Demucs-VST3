/*
  ==============================================================================

    This file contains the basic framework code for a JUCE plugin processor.

  ==============================================================================
*/

#pragma once
#include <JuceHeader.h>
#include <torch/script.h>
#include <torch/torch.h>

#include <vector>

//==============================================================================
/**
*/
class Demucs12AudioProcessor  : public juce::AudioProcessor
{
public:
    //==============================================================================
    Demucs12AudioProcessor();
    ~Demucs12AudioProcessor() override;

    //==============================================================================
    void prepareToPlay (double sampleRate, int samplesPerBlock) override;
    void releaseResources() override;
    
    /*
   #ifndef JucePlugin_PreferredChannelConfigurations
    bool isBusesLayoutSupported (const BusesLayout& layouts) const override;
   #endif
     */
    bool isBusesLayoutSupported (const BusesLayout& layouts) const override;
    
    
    void processBlock (juce::AudioBuffer<float>&, juce::MidiBuffer&) override;

    //==============================================================================
    juce::AudioProcessorEditor* createEditor() override;
    bool hasEditor() const override;

    //==============================================================================
    const juce::String getName() const override;

    bool acceptsMidi() const override;
    bool producesMidi() const override;
    bool isMidiEffect() const override;
    double getTailLengthSeconds() const override;

    //==============================================================================
    int getNumPrograms() override;
    int getCurrentProgram() override;
    void setCurrentProgram (int index) override;
    const juce::String getProgramName (int index) override;
    void changeProgramName (int index, const juce::String& newName) override;

    //==============================================================================
    void getStateInformation (juce::MemoryBlock& destData) override;
    void setStateInformation (const void* data, int sizeInBytes) override;

/*
private:
    bool loadModel();

    torch::jit::script::Module modelDemucs;
    torch::Device modelDevice = torch::Device(torch::kCPU);
    bool isModelLoaded = false;

    // Parámetros del modelo HDemucs definitivo
    static constexpr int modelChannels = 12;
    static constexpr int modelInputSamples = 4096;

    // Entrada multicanal del modelo.
    // Organización:
    //
    // canal 0 -> modelInput[0 * 4096 + i]
    // canal 1 -> modelInput[1 * 4096 + i]
    // ...
    // canal 11 -> modelInput[11 * 4096 + i]
    //
    std::vector<float> modelInput;
};
*/
    
private:
    void createPerfectHann(int overlapSamples);
    bool loadModel();
    
    torch::jit::script::Module modelDemucs;
    torch::Device modelDevice = torch::Device(torch::kCPU);
    bool isModelLoaded = false;
    
    // ============================================================
    // Parámetros del modelo HDemucs
    // ============================================================
    
    static constexpr int modelChannels = 12;
    
    // Ventana completa que entra al modelo.
    // Debe ser mayor que el bloque de REAPER para poder aplicar solape.
    
    //modelInoutSamples = 48000 muestras/s x 1.024 aqui sale el numero de muestras en un segundo como me ha pedido el pero vamos a probar con 8192 de momento que equivale a 17ms aprox
    static constexpr int modelInputSamples = 8192;
    
    // Número de muestras que se solapan entre bloques consecutivos.
    int overlapSamples = 512;
    
    // Historial multicanal para formar el tensor [1, 12, 8192]
    std::vector<float> modelInput;
    
    // Cola anterior de salida para solapamiento-suma.
    // Tamaño: 12 canales * overlapSamples.
    std::vector<float> prevTailOut;
    
    // Ventanas Hann para suavizar el solape.
    std::vector<float> winFirstHalf;
    std::vector<float> winSecondHalf;
    
    // Evita aplicar una cola inexistente en el primer bloque.
    bool hasPreviousTail = false;
    
    JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR (Demucs12AudioProcessor)
};
