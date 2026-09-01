#pragma once

#include <JuceHeader.h>
#include "PluginProcessor.h"

//==============================================================================
class Demucs12AudioProcessorEditor  : public juce::AudioProcessorEditor
{
public:
    explicit Demucs12AudioProcessorEditor (Demucs12AudioProcessor&);
    ~Demucs12AudioProcessorEditor() override;

    void paint (juce::Graphics&) override;
    void resized() override;

private:
    Demucs12AudioProcessor& audioProcessor;

    JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR (Demucs12AudioProcessorEditor)
};
