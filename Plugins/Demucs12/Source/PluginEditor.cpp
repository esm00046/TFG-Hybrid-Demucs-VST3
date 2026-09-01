#include "PluginProcessor.h"
#include "PluginEditor.h"

//==============================================================================
// Convierte texto UTF-8 a juce::String.
// Esto evita que los acentos aparezcan mal en REAPER/JUCE.
static juce::String txt(const char* text)
{
    return juce::String::fromUTF8(text);
}

//==============================================================================
Demucs12AudioProcessorEditor::Demucs12AudioProcessorEditor (Demucs12AudioProcessor& p)
    : AudioProcessorEditor (&p), audioProcessor (p)
{
    setSize (900, 520);

    // Permite que la ventana del plugin pueda redimensionarse.
    setResizable(true, true);
    setResizeLimits(760, 430, 1200, 720);
}

Demucs12AudioProcessorEditor::~Demucs12AudioProcessorEditor()
{
}

//==============================================================================
static void drawRoundedPanel(juce::Graphics& g,
                             juce::Rectangle<float> area,
                             juce::Colour fill,
                             juce::Colour border)
{
    g.setColour(fill);
    g.fillRoundedRectangle(area, 18.0f);

    g.setColour(border);
    g.drawRoundedRectangle(area, 18.0f, 1.4f);
}

static void drawSectionTitle(juce::Graphics& g,
                             const juce::String& title,
                             juce::Rectangle<int> area)
{
    g.setColour(juce::Colours::white);
    g.setFont(juce::Font(juce::FontOptions(17.0f).withStyle("Bold")));
    g.drawFittedText(title, area, juce::Justification::centredLeft, 1);
}

static void drawTextLine(juce::Graphics& g,
                         const juce::String& text,
                         int x,
                         int y,
                         int width,
                         int height,
                         float fontSize = 14.0f)
{
    g.setColour(juce::Colour::fromRGB(222, 228, 240));
    g.setFont(juce::Font(juce::FontOptions(fontSize)));
    g.drawFittedText(text, x, y, width, height, juce::Justification::centredLeft, 1);
}

static void drawTag(juce::Graphics& g,
                    const juce::String& text,
                    juce::Rectangle<float> area,
                    juce::Colour colour)
{
    g.setColour(colour.withAlpha(0.18f));
    g.fillRoundedRectangle(area, 8.0f);

    g.setColour(colour);
    g.drawRoundedRectangle(area, 8.0f, 1.2f);

    g.setFont(juce::Font(juce::FontOptions(12.0f).withStyle("Bold")));
    g.drawFittedText(text,
                     area.toNearestInt().reduced(8, 0),
                     juce::Justification::centred,
                     1);
}

//==============================================================================
void Demucs12AudioProcessorEditor::paint (juce::Graphics& g)
{
    g.fillAll(juce::Colour::fromRGB(12, 15, 22));

    auto bounds = getLocalBounds().reduced(24);

    // ============================================================
    // Cabecera principal
    // ============================================================

    auto header = bounds.removeFromTop(122).toFloat();

    juce::ColourGradient headerGradient(
        juce::Colour::fromRGB(24, 82, 135),
        header.getTopLeft(),
        juce::Colour::fromRGB(15, 28, 52),
        header.getBottomRight(),
        false
    );

    g.setGradientFill(headerGradient);
    g.fillRoundedRectangle(header, 20.0f);

    g.setColour(juce::Colour::fromRGB(90, 145, 205));
    g.drawRoundedRectangle(header, 20.0f, 1.5f);

    auto headerInt = header.toNearestInt().reduced(32, 20);

    g.setColour(juce::Colours::white);
    g.setFont(juce::Font(juce::FontOptions(23.0f).withStyle("Bold")));

    g.drawFittedText(
        txt(u8"De-bleeding en tiempo real de micrófonos cercanos mediante IA"),
        headerInt.removeFromTop(34),
        juce::Justification::centredLeft,
        1
    );

    g.setColour(juce::Colour::fromRGB(214, 224, 238));
    g.setFont(juce::Font(juce::FontOptions(15.0f).withStyle("Bold")));

    g.drawFittedText(
        txt(u8"Diseño e implementación de un VST3 multicanal"),
        headerInt.removeFromTop(26),
        juce::Justification::centredLeft,
        1
    );

    g.setColour(juce::Colour::fromRGB(180, 194, 214));
    g.setFont(juce::Font(juce::FontOptions(13.0f)));

    g.drawFittedText(
        txt(u8"Trabajo Fin de Grado · Procesamiento de audio en tiempo real · HDemucs + LibTorch"),
        headerInt.removeFromTop(22),
        juce::Justification::centredLeft,
        1
    );

    // Indicador visual
    auto indicator = juce::Rectangle<float>(header.getRight() - 125.0f,
                                            header.getY() + 42.0f,
                                            88.0f,
                                            30.0f);

    g.setColour(juce::Colour::fromRGB(20, 34, 48));
    g.fillRoundedRectangle(indicator, 15.0f);

    g.setColour(juce::Colour::fromRGB(85, 220, 145));
    g.fillEllipse(indicator.getRight() - 26.0f,
                  indicator.getY() + 7.0f,
                  15.0f,
                  15.0f);

    g.setColour(juce::Colour::fromRGB(205, 218, 232));
    g.setFont(juce::Font(juce::FontOptions(11.5f).withStyle("Bold")));
    g.drawText("READY",
               indicator.getX() + 10.0f,
               indicator.getY(),
               50.0f,
               indicator.getHeight(),
               juce::Justification::centredLeft);

    bounds.removeFromTop(24);

    // ============================================================
    // Paneles centrales adaptables
    // ============================================================

    auto centralArea = bounds.removeFromTop(bounds.getHeight() - 58);
    auto leftArea = centralArea.removeFromLeft(centralArea.getWidth() / 2 - 12);
    centralArea.removeFromLeft(24);
    auto rightArea = centralArea;

    auto leftPanel = leftArea.toFloat();
    auto rightPanel = rightArea.toFloat();

    drawRoundedPanel(g,
                     leftPanel,
                     juce::Colour::fromRGB(25, 29, 40),
                     juce::Colour::fromRGB(66, 75, 98));

    drawRoundedPanel(g,
                     rightPanel,
                     juce::Colour::fromRGB(25, 29, 40),
                     juce::Colour::fromRGB(66, 75, 98));

    // ============================================================
    // Panel izquierdo
    // ============================================================

    auto leftContent = leftArea.reduced(26, 24);

    drawSectionTitle(g,
                     txt(u8"Procesamiento implementado"),
                     leftContent.removeFromTop(30));

    leftContent.removeFromTop(12);

    const int lineHeight = 27;
    const int leftX = leftContent.getX();
    int y = leftContent.getY();
    const int leftW = leftContent.getWidth();

    drawTextLine(g, txt(u8"- Modelo HDemucs exportado a TorchScript"), leftX, y, leftW, lineHeight); y += lineHeight;
    drawTextLine(g, txt(u8"- Integración en JUCE mediante LibTorch"), leftX, y, leftW, lineHeight); y += lineHeight;
    drawTextLine(g, txt(u8"- Ejecución acelerada mediante MPS / Apple Metal"), leftX, y, leftW, lineHeight); y += lineHeight;
    drawTextLine(g, txt(u8"- Procesamiento multicanal en tiempo real"), leftX, y, leftW, lineHeight); y += lineHeight;
    drawTextLine(g, txt(u8"- Gestión flexible de canales reales"), leftX, y, leftW, lineHeight); y += lineHeight;
    drawTextLine(g, txt(u8"- Relleno con ceros de canales no utilizados"), leftX, y, leftW, lineHeight); y += lineHeight;
    drawTextLine(g, txt(u8"- Solapamiento-suma para suavizar bloques"), leftX, y, leftW, lineHeight); y += lineHeight;

    // Etiquetas decorativas inferiores
    auto tagAreaLeft = leftArea.reduced(26, 24).removeFromBottom(34).toFloat();

    drawTag(g,
            txt(u8"HDemucs"),
            tagAreaLeft.removeFromLeft(90),
            juce::Colour::fromRGB(100, 175, 240));

    tagAreaLeft.removeFromLeft(10);

    drawTag(g,
            txt(u8"TorchScript"),
            tagAreaLeft.removeFromLeft(118),
            juce::Colour::fromRGB(100, 220, 170));

    tagAreaLeft.removeFromLeft(10);

    drawTag(g,
            txt(u8"LibTorch"),
            tagAreaLeft.removeFromLeft(92),
            juce::Colour::fromRGB(220, 180, 90));

    // ============================================================
    // Panel derecho
    // ============================================================

    auto rightContent = rightArea.reduced(26, 24);

    drawSectionTitle(g,
                     txt(u8"Configuración del prototipo"),
                     rightContent.removeFromTop(30));

    rightContent.removeFromTop(12);

    const int rightX = rightContent.getX();
    int ry = rightContent.getY();
    const int rightW = rightContent.getWidth();

    drawTextLine(g, txt(u8"Canales internos del modelo: 12"), rightX, ry, rightW, lineHeight); ry += lineHeight;
    drawTextLine(g, txt(u8"Ventana de contexto: 8192 muestras"), rightX, ry, rightW, lineHeight); ry += lineHeight;
    drawTextLine(g, txt(u8"Solapamiento-suma: 512 muestras"), rightX, ry, rightW, lineHeight); ry += lineHeight;
    drawTextLine(g, txt(u8"Frecuencia de trabajo: 48 kHz"), rightX, ry, rightW, lineHeight); ry += lineHeight;
    drawTextLine(g, txt(u8"Bloque habitual de REAPER: 4096 muestras"), rightX, ry, rightW, lineHeight); ry += lineHeight;
    drawTextLine(g, txt(u8"Entrada variable: 1 a 12 canales reales"), rightX, ry, rightW, lineHeight); ry += lineHeight;
    drawTextLine(g, txt(u8"Salida redirigida mediante bus multicanal"), rightX, ry, rightW, lineHeight); ry += lineHeight;

    auto tagAreaRight = rightArea.reduced(26, 24).removeFromBottom(34).toFloat();

    drawTag(g,
            txt(u8"VST3"),
            tagAreaRight.removeFromLeft(72),
            juce::Colour::fromRGB(120, 180, 255));

    tagAreaRight.removeFromLeft(10);

    drawTag(g,
            txt(u8"12 canales"),
            tagAreaRight.removeFromLeft(104),
            juce::Colour::fromRGB(95, 215, 145));

    tagAreaRight.removeFromLeft(10);

    drawTag(g,
            txt(u8"MPS / Metal"),
            tagAreaRight.removeFromLeft(112),
            juce::Colour::fromRGB(220, 180, 100));

    // ============================================================
    // Pie inferior
    // ============================================================

    bounds.removeFromTop(18);

    auto footer = bounds.removeFromTop(34).toFloat();

    g.setColour(juce::Colour::fromRGB(22, 26, 36));
    g.fillRoundedRectangle(footer, 10.0f);

    g.setColour(juce::Colour::fromRGB(165, 176, 194));
    g.setFont(juce::Font(juce::FontOptions(13.0f)));

    g.drawFittedText(
        txt(u8"Flujo recomendado: pistas seleccionadas en REAPER -> ReaScript -> Demucs12 Bus -> pistas separadas de salida"),
        footer.toNearestInt().reduced(16, 0),
        juce::Justification::centredLeft,
        1
    );
}

void Demucs12AudioProcessorEditor::resized()
{
    // La interfaz se dibuja directamente en paint().
    // Al redimensionar la ventana, JUCE llama de nuevo a paint()
    // y los paneles se recalculan proporcionalmente.
}
