------------------------------------------------------------
-- Demucs12 Source Separation Bus
------------------------------------------------------------
--
-- Este script automatiza el ruteo en REAPER para usar el
-- plugin Demucs12.
--
-- Funcionamiento:
--
-- 1. El usuario selecciona las pistas que quiere procesar.
-- 2. El script crea una pista bus multicanal.
-- 3. Envía cada pista/canal seleccionado al bus.
-- 4. Inserta el plugin Demucs12 en el bus.
-- 5. Crea una pista independiente por cada canal de salida.
-- 6. Envía cada canal de salida del bus a su pista separada.
--
-- El plugin Demucs12 puede recibir menos de 12 canales reales.
-- Internamente rellenará los canales restantes con ceros.
--
------------------------------------------------------------

-- Nombre del plugin tal y como aparece en REAPER.
-- Si no lo encuentra, prueba cambiando esta línea por:
-- local pluginName = "Demucs12"
-- local pluginName = "VST3: Demucs12"
-- local pluginName = "VST3: Demucs12 (yourcompany)"
local pluginName = "VST3: Demucs12 (yourcompany)"

------------------------------------------------------------
-- Información de pistas/canales seleccionados
------------------------------------------------------------

local tracks = {}
local sources = {}
local totalChannels = 0

local MAX_DEMUCS_CHANNELS = 12

function updateSelectedTracksInfo()
    tracks = {}
    sources = {}
    totalChannels = 0

    local nTracks = reaper.CountSelectedTracks(0)

    for i = 0, nTracks - 1 do
        local track = reaper.GetSelectedTrack(0, i)
        local _, name = reaper.GetTrackName(track)

        local nchan = reaper.GetMediaTrackInfo_Value(track, "I_NCHAN")

        -- En REAPER, una pista normal suele aparecer como 2 canales,
        -- incluso aunque el archivo sea mono.
        --
        -- Para el caso habitual de pistas WAV independientes,
        -- tratamos una pista estéreo normal como una única fuente.
        --
        -- Si la pista tiene más de 2 canales, se trata como pista
        -- multicanal y cada canal se considera una fuente independiente.
        local validchan = nchan

        if nchan == 2 then
            validchan = 1
        end

        table.insert(tracks,
        {
            track = track,
            name = name,
            channels = nchan,
            validchannels = validchan
        })

        if nchan == 2 then
            -- Fuente única procedente de la pista.
            table.insert(sources,
            {
                track = track,
                trackName = name,
                label = name,
                sourceChannel = 0
            })

            totalChannels = totalChannels + 1
        else
            -- Pista multicanal: cada canal se trata como una fuente.
            for ch = 0, nchan - 1 do
                table.insert(sources,
                {
                    track = track,
                    trackName = name,
                    label = name .. " Ch " .. tostring(ch + 1),
                    sourceChannel = ch
                })

                totalChannels = totalChannels + 1
            end
        end
    end
end

------------------------------------------------------------
-- Interfaz
------------------------------------------------------------

updateSelectedTracksInfo()

local WIDTH = 560
local HEIGHT = 620 + 20 * math.max(#sources - 1, 0)

gfx.init("Demucs12 Debleeding Bus", WIDTH, HEIGHT)

------------------------------------------------------------

local function draw_panel(x, y, w, h)
    gfx.set(0.82, 0.82, 0.82)
    gfx.rect(x, y, w, h, 1)
    gfx.set(0.60, 0.60, 0.60)
    gfx.rect(x, y, w, h, 0)
end

------------------------------------------------------------

local function draw_button(bx, by, bw, bh, text)
    -- sombra
    gfx.set(0.45, 0.45, 0.45)
    gfx.rect(bx + 2, by + 2, bw, bh, 1)

    -- botón
    gfx.set(0.18, 0.45, 0.82)
    gfx.rect(bx, by, bw, bh, 1)

    -- borde
    gfx.set(1, 1, 1)
    gfx.rect(bx, by, bw, bh, 0)

    -- texto
    gfx.set(1, 1, 1)
    gfx.x = bx + 28
    gfx.y = by + 10
    gfx.drawstr(text)
end

------------------------------------------------------------

local function drawSourceTable(x, y, width, sources)
    local rowH = 22
    local headerH = 24
    local titleH = 22

    gfx.setfont(2, "Arial", 14)
    gfx.set(0, 0, 0)
    gfx.x = x
    gfx.y = y
    gfx.drawstr("Input Channel Distribution")
    y = y + titleH

    -- Header
    gfx.set(0.25, 0.35, 0.55)
    gfx.rect(x, y, width, headerH, 1)

    gfx.set(1, 1, 1)
    gfx.setfont(2, "Arial", 13)

    gfx.x = x + 10
    gfx.y = y + 4
    gfx.drawstr("Source")

    gfx.x = x + width - 95
    gfx.drawstr("Bus ch.")

    -- Rows
    local rowY = y + headerH

    for i, s in ipairs(sources) do
        if (i % 2 == 0) then
            gfx.set(0.28, 0.28, 0.28)
        else
            gfx.set(0.22, 0.22, 0.22)
        end

        gfx.rect(x, rowY, width, rowH, 1)

        gfx.set(0.45, 0.45, 0.45)
        gfx.rect(x, rowY, width, rowH, 0)

        gfx.set(1, 1, 1)
        gfx.x = x + 10
        gfx.y = rowY + 4
        gfx.drawstr(s.label)

        local txt = tostring(i)
        local tw = gfx.measurestr(txt)

        gfx.x = x + width - tw - 25
        gfx.y = rowY + 4
        gfx.drawstr(txt)

        rowY = rowY + rowH
    end

    return titleH + headerH + #sources * rowH
end

------------------------------------------------------------

local function draw()
    gfx.set(0.92, 0.92, 0.92)
    gfx.rect(0, 0, gfx.w, gfx.h, 1)

    -- Header
    gfx.set(0.18, 0.42, 0.75)
    gfx.rect(0, 0, gfx.w, 42, 1)

    gfx.set(1, 1, 1)
    gfx.setfont(1, "Arial", 18)
    gfx.x = 18
    gfx.y = 10
    gfx.drawstr("Demucs12 Source Separation Bus")

    -- Panels
    draw_panel(15, 55, gfx.w - 30, 70)
    draw_panel(15, 135, gfx.w - 30, 70)
    draw_panel(15, 215, gfx.w - 30, gfx.h - 290)

    -- Description
    gfx.set(0, 0, 0)
    gfx.setfont(2, "Arial", 14)

    gfx.x = 25
    gfx.y = 62
    gfx.drawstr("This script creates a multichannel bus for Demucs12 debleeding.")

    gfx.x = 25
    gfx.y = 84
    gfx.drawstr("Selected tracks are routed into a bus, processed, and separated")

    gfx.x = 25
    gfx.y = 104
    gfx.drawstr("again into independent output tracks.")

    -- Summary
    gfx.set(0, 0, 0)
    gfx.y = 145
    gfx.x = 25
    gfx.drawstr("Selected tracks:")
    gfx.x = 210
    gfx.drawstr(tostring(#tracks))

    gfx.y = 168
    gfx.x = 25
    gfx.drawstr("Total real sources/channels:")
    gfx.x = 210
    gfx.drawstr(tostring(totalChannels))

    gfx.y = 190
    gfx.x = 25

    if totalChannels > MAX_DEMUCS_CHANNELS then
        gfx.set(0.85, 0.05, 0.05)
        gfx.drawstr("ERROR: Demucs12 supports a maximum of 12 real channels.")
    elseif totalChannels == 0 then
        gfx.set(0.85, 0.05, 0.05)
        gfx.drawstr("ERROR: select at least one track before running the script.")
    else
        gfx.set(0.0, 0.45, 0.0)
        gfx.drawstr("Ready. Missing channels up to 12 will be filled with zeros by the plugin.")
    end

    -- Table
    drawSourceTable(25, 225, gfx.w - 50, sources)

    -- Button
    local bx = gfx.w / 2 - 80
    local by = gfx.h - 55
    local bw = 160
    local bh = 35

    draw_button(bx, by, bw, bh, "RUN SCRIPT")

    return bx, by, bw, bh
end

------------------------------------------------------------
-- Crear bus y ruteo
------------------------------------------------------------

local function execute()
    updateSelectedTracksInfo()

    if totalChannels == 0 then
        reaper.ShowMessageBox("Select at least one track before running the script.",
                              "Demucs12", 0)
        return
    end

    if totalChannels > MAX_DEMUCS_CHANNELS then
        reaper.ShowMessageBox("Too many channels selected. Demucs12 supports a maximum of 12 real channels.",
                              "Demucs12", 0)
        return
    end

    reaper.ShowConsoleMsg("Creating Demucs12 bus...\n")

    reaper.Undo_BeginBlock()

    ------------------------------------------------------------
    -- 1. Crear pista bus
    ------------------------------------------------------------

    local insertIndex = reaper.CountTracks(0)

    reaper.InsertTrackAtIndex(insertIndex, true)
    local bus = reaper.GetTrack(0, insertIndex)

    reaper.GetSetMediaTrackInfo_String(bus, "P_NAME", "Demucs12 Bus", true)

    -- El bus tendrá tantos canales reales como fuentes seleccionadas.
    -- El plugin rellenará internamente con ceros hasta llegar a 12.
    reaper.SetMediaTrackInfo_Value(bus, "I_NCHAN", totalChannels)

    ------------------------------------------------------------
    -- 2. Insertar plugin Demucs12 en el bus
    ------------------------------------------------------------

    local fxIndex = reaper.TrackFX_AddByName(bus, pluginName, false, -1)

    if fxIndex < 0 then
        reaper.ShowMessageBox("The plugin was not found automatically.\n\nCheck the plugin name in the script:\n" .. pluginName,
                              "Demucs12 plugin not found", 0)
    else
        reaper.ShowConsoleMsg("Inserted plugin: " .. pluginName .. "\n")
    end

    ------------------------------------------------------------
    -- 3. Crear pistas de salida
    ------------------------------------------------------------

    local output = {}

    for i, s in ipairs(sources) do
        local idx = insertIndex + i

        reaper.InsertTrackAtIndex(idx, true)
        local outTrack = reaper.GetTrack(0, idx)

        reaper.GetSetMediaTrackInfo_String(outTrack,
                                        "P_NAME",
                                        "Separated " .. tostring(i) .. " - " .. s.label,
                                        true)

        -- Pista estéreo normal para monitorizar/procesar la salida.
        reaper.SetMediaTrackInfo_Value(outTrack, "I_NCHAN", 2)

        output[i] = outTrack
    end

    ------------------------------------------------------------
    -- 4. Envíos desde pistas originales al bus
    ------------------------------------------------------------
    --
    -- 1024 indica envío mono de un canal concreto.
    --
    -- Cada fuente se envía a un canal diferente del bus:
    --
    --      fuente 1 -> bus canal 1
    --      fuente 2 -> bus canal 2
    --      ...
    --

    for i, s in ipairs(sources) do
        local src = s.track

        local send = reaper.CreateTrackSend(src, bus)

        -- Canal de origen dentro de la pista seleccionada.
        reaper.SetTrackSendInfo_Value(src, 0, send, "I_SRCCHAN", s.sourceChannel + 1024)

        -- Canal de destino dentro del bus.
        reaper.SetTrackSendInfo_Value(src, 0, send, "I_DSTCHAN", (i - 1) + 1024)

        reaper.ShowConsoleMsg("Input: " .. s.label .. " -> Bus channel " .. tostring(i) .. "\n")
    end

    ------------------------------------------------------------
    -- 5. Envíos desde el bus a pistas de salida
    ------------------------------------------------------------
    --
    -- Cada canal de salida del bus se envía a una pista separada.
    --

    for i, outTrack in ipairs(output) do
        local send = reaper.CreateTrackSend(bus, outTrack)

        -- Tomamos el canal i del bus.
        reaper.SetTrackSendInfo_Value(bus, 0, send, "I_SRCCHAN", (i - 1) + 1024)

        -- Lo enviamos a la entrada principal de la pista destino.
        reaper.SetTrackSendInfo_Value(bus, 0, send, "I_DSTCHAN", 0)

        reaper.ShowConsoleMsg("Output: Bus channel " .. tostring(i) .. " -> " .. sources[i].label .. " separated track\n")
    end

    ------------------------------------------------------------
    -- 6. Desactivar master send de pistas originales y bus
    ------------------------------------------------------------

    for _, t in ipairs(tracks) do
        reaper.SetMediaTrackInfo_Value(t.track, "B_MAINSEND", 0)
    end

    reaper.SetMediaTrackInfo_Value(bus, "B_MAINSEND", 0)

    ------------------------------------------------------------
    -- 7. Actualizar REAPER
    ------------------------------------------------------------

    reaper.TrackList_AdjustWindows(false)
    reaper.UpdateArrange()

    reaper.Undo_EndBlock("Create Demucs12 Source Separation Bus", -1)

    reaper.ShowConsoleMsg("Demucs12 routing completed.\n")
end

------------------------------------------------------------
-- Bucle principal
------------------------------------------------------------

function loop()
    updateSelectedTracksInfo()

    local bx, by, bw, bh = draw()

    local mx, my = gfx.mouse_x, gfx.mouse_y

    if gfx.mouse_cap == 1 then
        if mx >= bx and mx <= bx + bw and
           my >= by and my <= by + bh then
            execute()
            gfx.quit()
            return
        end
    end

    if gfx.getchar() >= 0 then
        reaper.defer(loop)
    end
end

loop()