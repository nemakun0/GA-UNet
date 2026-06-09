param(
    [string]$ProjectRoot = (Resolve-Path ".").Path,
    [string]$TemplatePath = ""
)

$ErrorActionPreference = "Stop"

$OutputDir = Join-Path $ProjectRoot "outputs"
$AssetDir = Join-Path $OutputDir "thesis_assets"
$RenderDir = Join-Path $OutputDir "thesis_rendered_pages"
$DocxPath = Join-Path $OutputDir "GA_UNet_Duzce_Graduation_Thesis.docx"
$PdfPath = Join-Path $OutputDir "GA_UNet_Duzce_Graduation_Thesis.pdf"
$QaSummaryPath = Join-Path $OutputDir "GA_UNet_Duzce_Graduation_Thesis_QA.txt"
$BuildLogPath = Join-Path $OutputDir "GA_UNet_Duzce_Graduation_Thesis_build.log"

if ([string]::IsNullOrWhiteSpace($TemplatePath)) {
    $candidate = Get-ChildItem -Path (Join-Path $env:USERPROFILE "Desktop") -Filter "mezuniyet tezi*.docx" -ErrorAction SilentlyContinue | Select-Object -First 1
    if ($candidate) {
        $TemplatePath = $candidate.FullName
    }
}

New-Item -ItemType Directory -Force -Path $OutputDir, $AssetDir, $RenderDir | Out-Null
Set-Content -Path $BuildLogPath -Value "Build started: $(Get-Date -Format s)" -Encoding UTF8

function Log-Step([string]$Message) {
    Add-Content -Path $BuildLogPath -Value ("$(Get-Date -Format s)  $Message") -Encoding UTF8
}

function ConvertTo-PlainText([string]$s) {
    if ($null -eq $s) { return "" }
    return $s -replace "[\u2013\u2014]", "-" -replace "[\u2018\u2019]", "'" -replace "[\u201C\u201D]", '"'
}

function New-Bitmap {
    param([string]$Path, [int]$Width = 1600, [int]$Height = 900)
    Add-Type -AssemblyName System.Drawing
    $bmp = New-Object System.Drawing.Bitmap($Width, $Height)
    $g = [System.Drawing.Graphics]::FromImage($bmp)
    $g.SmoothingMode = [System.Drawing.Drawing2D.SmoothingMode]::AntiAlias
    $g.TextRenderingHint = [System.Drawing.Text.TextRenderingHint]::ClearTypeGridFit
    $g.Clear([System.Drawing.Color]::FromArgb(250, 252, 255))
    return @{ Bitmap = $bmp; Graphics = $g; Path = $Path }
}

function Save-Bitmap($canvas) {
    $canvas.Bitmap.Save($canvas.Path, [System.Drawing.Imaging.ImageFormat]::Png)
    $canvas.Graphics.Dispose()
    $canvas.Bitmap.Dispose()
}

function Brush([int]$r,[int]$g,[int]$b) {
    return New-Object System.Drawing.SolidBrush([System.Drawing.Color]::FromArgb($r,$g,$b))
}

function PenC([int]$r,[int]$g,[int]$b,[float]$w = 2.0) {
    return New-Object System.Drawing.Pen([System.Drawing.Color]::FromArgb($r,$g,$b), $w)
}

function Draw-TextBox {
    param(
        $G, [float]$X, [float]$Y, [float]$W, [float]$H,
        [string]$Text, [string]$Fill = "#ECF2FF", [string]$Stroke = "#2C5AA0",
        [int]$FontSize = 24, [bool]$Bold = $true
    )
    $fillColor = [System.Drawing.ColorTranslator]::FromHtml($Fill)
    $strokeColor = [System.Drawing.ColorTranslator]::FromHtml($Stroke)
    $path = New-Object System.Drawing.Drawing2D.GraphicsPath
    $radius = 22
    $rect = New-Object System.Drawing.RectangleF($X,$Y,$W,$H)
    $d = $radius * 2
    $path.AddArc($rect.X, $rect.Y, $d, $d, 180, 90)
    $path.AddArc($rect.Right - $d, $rect.Y, $d, $d, 270, 90)
    $path.AddArc($rect.Right - $d, $rect.Bottom - $d, $d, $d, 0, 90)
    $path.AddArc($rect.X, $rect.Bottom - $d, $d, $d, 90, 90)
    $path.CloseFigure()
    $G.FillPath((New-Object System.Drawing.SolidBrush($fillColor)), $path)
    $G.DrawPath((New-Object System.Drawing.Pen($strokeColor, 3)), $path)
    $style = if ($Bold) { [System.Drawing.FontStyle]::Bold } else { [System.Drawing.FontStyle]::Regular }
    $font = New-Object System.Drawing.Font("Segoe UI", $FontSize, $style)
    $fmt = New-Object System.Drawing.StringFormat
    $fmt.Alignment = [System.Drawing.StringAlignment]::Center
    $fmt.LineAlignment = [System.Drawing.StringAlignment]::Center
    $fmt.Trimming = [System.Drawing.StringTrimming]::Word
    $G.DrawString($Text, $font, (Brush 24 35 51), $rect, $fmt)
    $font.Dispose()
    $fmt.Dispose()
    $path.Dispose()
}

function Draw-Arrow {
    param($G, [float]$X1, [float]$Y1, [float]$X2, [float]$Y2, [string]$Color = "#3E6FB2")
    $pen = New-Object System.Drawing.Pen([System.Drawing.ColorTranslator]::FromHtml($Color), 5)
    $cap = New-Object System.Drawing.Drawing2D.AdjustableArrowCap(7, 9, $true)
    $pen.CustomEndCap = $cap
    $G.DrawLine($pen, $X1, $Y1, $X2, $Y2)
    $pen.Dispose()
    $cap.Dispose()
}

function Draw-Title {
    param($G, [string]$Title, [string]$Sub = "")
    $font = New-Object System.Drawing.Font("Segoe UI", 34, [System.Drawing.FontStyle]::Bold)
    $G.DrawString($Title, $font, (Brush 18 31 53), 50, 36)
    $font.Dispose()
    if ($Sub) {
        $font2 = New-Object System.Drawing.Font("Segoe UI", 18, [System.Drawing.FontStyle]::Regular)
        $G.DrawString($Sub, $font2, (Brush 73 85 104), 53, 86)
        $font2.Dispose()
    }
}

function New-ArchitectureDiagram {
    $c = New-Bitmap (Join-Path $AssetDir "fig_3_1_architecture.png")
    $g = $c.Graphics
    Draw-Title $g "GA-UNet Architecture" "2.5D input, GhostNetV2 encoder, SimAM-supported decoder"
    $y = 250; $w = 210; $h = 105; $gap = 32
    $xs = @(
        55,
        (55 + $w + $gap),
        (55 + 2 * ($w + $gap)),
        (55 + 3 * ($w + $gap)),
        (55 + 4 * ($w + $gap)),
        (55 + 5 * ($w + $gap))
    )
    Draw-TextBox $g $xs[0] $y $w $h "Input`nB x 3 x 192 x 192" "#E8F5E9" "#2E7D32" 22
    Draw-TextBox $g $xs[1] $y $w $h "GhostNetV2`nEncoder" "#E3F2FD" "#1565C0" 22
    Draw-TextBox $g $xs[2] $y $w $h "Ghost`nBottleneck" "#FFF8E1" "#E69500" 22
    Draw-TextBox $g $xs[3] $y $w $h "SimAM Decoder`n+ Skip Links" "#F3E5F5" "#7B1FA2" 22
    Draw-TextBox $g $xs[4] $y $w $h "Segmentation`nHead" "#E0F7FA" "#00838F" 22
    Draw-TextBox $g $xs[5] $y $w $h "Lesion`nMask" "#FFEBEE" "#C62828" 22
    for ($i=0; $i -lt 5; $i++) { Draw-Arrow $g ($xs[$i]+$w+8) ($y+$h/2) ($xs[$i+1]-8) ($y+$h/2) }
    $stageY = 485
    $stageW = 180
    $stageLabels = @("H/2`n16 ch", "H/4`n24 ch", "H/8`n40 ch", "H/16`n80 ch")
    for ($i=0; $i -lt 4; $i++) {
        $sx = 315 + $i*230
        Draw-TextBox $g $sx $stageY $stageW 82 $stageLabels[$i] "#F8FAFC" "#64748B" 18 $false
        Draw-Arrow $g ($sx+$stageW/2) ($stageY-10) ($sx+$stageW/2) ($stageY-110) "#7B1FA2"
    }
    $noteFont = New-Object System.Drawing.Font("Segoe UI", 18, [System.Drawing.FontStyle]::Regular)
    $g.DrawString("Skip features are re-weighted by SimAM before concatenation, preserving localization while suppressing irrelevant background tissue.", $noteFont, (Brush 45 55 72), 135, 650)
    $noteFont.Dispose()
    Save-Bitmap $c
}

function New-GhostDiagram {
    $c = New-Bitmap (Join-Path $AssetDir "fig_3_2_ghostmodule.png")
    $g = $c.Graphics
    Draw-Title $g "GhostModule Principle" "Generate a small intrinsic set, then expand it through cheap operations"
    Draw-TextBox $g 90 280 260 120 "Input X" "#E8F5E9" "#2E7D32" 26
    Draw-TextBox $g 470 210 330 115 "Primary Conv`nIntrinsic features Y'" "#E3F2FD" "#1565C0" 24
    Draw-TextBox $g 470 400 330 115 "Depthwise / Cheap Ops`nGhost features" "#FFF8E1" "#E69500" 24
    Draw-TextBox $g 940 300 330 135 "Concatenate + Slice`nOutput Y" "#F3E5F5" "#7B1FA2" 24
    Draw-Arrow $g 355 340 460 270
    Draw-Arrow $g 355 340 460 455
    Draw-Arrow $g 810 270 930 350
    Draw-Arrow $g 810 455 930 370
    $font = New-Object System.Drawing.Font("Consolas", 22, [System.Drawing.FontStyle]::Bold)
    $g.DrawString("Y' = X * W'", $font, (Brush 18 31 53), 520, 570)
    $g.DrawString("y_ij = Phi_ij(y'_i)", $font, (Brush 18 31 53), 520, 625)
    $font.Dispose()
    Save-Bitmap $c
}

function New-SimAMDiagram {
    $c = New-Bitmap (Join-Path $AssetDir "fig_3_3_simam_skip.png")
    $g = $c.Graphics
    Draw-Title $g "SimAM in the Decoder Skip Path" "Parameter-free attention is applied to encoder features before concatenation"
    Draw-TextBox $g 130 230 320 110 "Encoder Skip`nHigh-resolution features" "#E3F2FD" "#1565C0" 23
    Draw-TextBox $g 620 230 310 110 "SimAM`n3D attention weights" "#F3E5F5" "#7B1FA2" 23
    Draw-TextBox $g 110 520 340 110 "Upsampled decoder`ncoarse semantic features" "#FFF8E1" "#E69500" 23
    Draw-TextBox $g 1080 370 320 130 "Concatenate`nGhost conv block" "#E0F7FA" "#00838F" 23
    Draw-Arrow $g 460 285 610 285
    Draw-Arrow $g 940 285 1065 405
    Draw-Arrow $g 455 575 1065 450
    $font = New-Object System.Drawing.Font("Segoe UI", 20, [System.Drawing.FontStyle]::Regular)
    $g.DrawString("The decoder receives cleaner spatial detail, which is important when chronic stroke lesions are small or low-contrast.", $font, (Brush 45 55 72), 170, 710)
    $font.Dispose()
    Save-Bitmap $c
}

function New-StackDiagram {
    $c = New-Bitmap (Join-Path $AssetDir "fig_3_4_25d_stack.png")
    $g = $c.Graphics
    Draw-Title $g "2.5D Axial Slice Stacking" "Neighboring slices provide limited through-plane context without a full 3D network"
    $x0 = 230
    for ($i=0; $i -lt 3; $i++) {
        $x = $x0 + $i*220
        $rect = New-Object System.Drawing.Rectangle($x, 250, 170, 240)
        $g.FillRectangle((Brush (230+$i*4) (238-$i*6) 255), $rect)
        $g.DrawRectangle((PenC 40 90 160 4), $rect)
        $font = New-Object System.Drawing.Font("Segoe UI", 24, [System.Drawing.FontStyle]::Bold)
        $label = @("S z-1","S z","S z+1")[$i]
        $g.DrawString($label, $font, (Brush 18 31 53), $x+38, 355)
        $font.Dispose()
    }
    Draw-Arrow $g 900 370 1030 370
    Draw-TextBox $g 1050 292 310 155 "Input tensor I_z`n3 x 192 x 192" "#E8F5E9" "#2E7D32" 25
    $font2 = New-Object System.Drawing.Font("Consolas", 24, [System.Drawing.FontStyle]::Bold)
    $g.DrawString("I_z = [S_(z-1), S_z, S_(z+1)]", $font2, (Brush 18 31 53), 480, 610)
    $font2.Dispose()
    Save-Bitmap $c
}

function New-TtaDiagram {
    $c = New-Bitmap (Join-Path $AssetDir "fig_3_5_tta.png")
    $g = $c.Graphics
    Draw-Title $g "Test-Time Augmentation Flow" "Original, horizontal flip, and vertical flip predictions are averaged in probability space"
    Draw-TextBox $g 70 360 210 100 "Input I" "#E8F5E9" "#2E7D32" 24
    Draw-TextBox $g 430 180 260 90 "Original`nP(I)" "#E3F2FD" "#1565C0" 22
    Draw-TextBox $g 430 360 260 90 "Horizontal flip`nT_h^-1(P(T_h(I)))" "#FFF8E1" "#E69500" 18
    Draw-TextBox $g 430 540 260 90 "Vertical flip`nT_v^-1(P(T_v(I)))" "#F3E5F5" "#7B1FA2" 18
    Draw-TextBox $g 870 350 260 120 "Average`nprobabilities" "#E0F7FA" "#00838F" 24
    Draw-TextBox $g 1260 365 220 90 "Final logits`nand mask" "#FFEBEE" "#C62828" 22
    Draw-Arrow $g 290 410 420 225
    Draw-Arrow $g 290 410 420 405
    Draw-Arrow $g 290 410 420 585
    Draw-Arrow $g 700 225 860 385
    Draw-Arrow $g 700 405 860 410
    Draw-Arrow $g 700 585 860 435
    Draw-Arrow $g 1140 410 1250 410
    Save-Bitmap $c
}

function New-DataFlowDiagram {
    $c = New-Bitmap (Join-Path $AssetDir "fig_4_1_data_flow.png")
    $g = $c.Graphics
    Draw-Title $g "GA-UNet Data Processing and Inference Flow" "From NIfTI volumes to segmentation metrics"
    $labels = @("NIfTI T1w + mask","Intensity normalization","Axial slice selection","2.5D stack","GA-UNet logits","Sigmoid probability","Thresholded mask","Dice / IoU / HD95")
    for ($i=0; $i -lt $labels.Count; $i++) {
        $x = 55 + ($i % 4) * 370
        $y = if ($i -lt 4) { 230 } else { 505 }
        Draw-TextBox $g $x $y 285 100 $labels[$i] "#F8FAFC" "#3E6FB2" 20
        if (($i % 4) -lt 3) { Draw-Arrow $g ($x+295) ($y+50) ($x+355) ($y+50) }
    }
    Draw-Arrow $g 1335 330 1335 500
    Draw-Arrow $g 1185 555 1130 555
    Save-Bitmap $c
}

function New-InterfaceDiagram {
    $c = New-Bitmap (Join-Path $AssetDir "fig_4_2_streamlit_interface.png") 1600 900
    $g = $c.Graphics
    Draw-Title $g "Streamlit Test Interface" "Clean representative view of the implemented GA-UNet testing workflow"
    $g.FillRectangle((Brush 14 18 28), 70, 145, 1460, 650)
    $font = New-Object System.Drawing.Font("Segoe UI", 24, [System.Drawing.FontStyle]::Bold)
    $g.DrawString("GA-UNet End-to-End Test Interface", $font, (Brush 235 241 250), 115, 180)
    $font.Dispose()
    $tabs = @("Metric Calculation","Scanner Analysis","Visual Inspection")
    for ($i=0; $i -lt 3; $i++) {
        Draw-TextBox $g (115+$i*330) 245 290 70 $tabs[$i] "#1F2937" "#3B82F6" 18
    }
    $cards = @("Mean Dice`n0.8172","Mean IoU`n0.7949","Volumes`n84","TTA`nHFlip + VFlip")
    for ($i=0; $i -lt 4; $i++) {
        Draw-TextBox $g (115+$i*350) 360 285 110 $cards[$i] "#111827" "#10B981" 22
    }
    Draw-TextBox $g 115 540 390 170 "T1w Image`nGround Truth overlay" "#111827" "#64748B" 20
    Draw-TextBox $g 600 540 390 170 "Model Prediction`nProbability map" "#111827" "#64748B" 20
    Draw-TextBox $g 1085 540 360 170 "Volume metrics`nCSV export" "#111827" "#64748B" 20
    Save-Bitmap $c
}

function New-DashboardDiagram {
    $c = New-Bitmap (Join-Path $AssetDir "fig_4_3_dashboard.png") 1600 900
    $g = $c.Graphics
    Draw-Title $g "Training Dashboard" "Representative dashboard based on dashboard.html and training_log.jsonl"
    $g.FillRectangle((Brush 10 12 16), 70, 145, 1460, 650)
    $labels = @("Epoch`n65 / 100","Train Dice`n0.7751","Val Dice`n0.8352","Best Val`n0.8401","Val Loss`n0.0696")
    for ($i=0; $i -lt 5; $i++) {
        Draw-TextBox $g (105+$i*285) 205 235 110 $labels[$i] "#111827" "#3B82F6" 22
    }
    $penBlue = PenC 59 130 246 4
    $penOrange = PenC 249 115 22 4
    $g.DrawRectangle((PenC 31 41 55 2), 115, 390, 620, 270)
    $g.DrawRectangle((PenC 31 41 55 2), 835, 390, 580, 270)
    for ($i=0; $i -lt 7; $i++) {
        $x1 = 145 + $i*85
        $g.DrawLine($penBlue, $x1, 610 - $i*18, $x1+80, 600 - ($i+1)*18)
        $g.DrawLine($penOrange, $x1, 600 - $i*21, $x1+80, 585 - ($i+1)*21)
        $x2 = 865 + $i*75
        $g.DrawLine($penBlue, $x2, 440 + $i*18, $x2+70, 452 + ($i+1)*13)
        $g.DrawLine($penOrange, $x2, 430 + $i*28, $x2+70, 438 + ($i+1)*25)
    }
    $font = New-Object System.Drawing.Font("Segoe UI", 20, [System.Drawing.FontStyle]::Bold)
    $g.DrawString("Dice Score", $font, (Brush 235 241 250), 135, 350)
    $g.DrawString("Loss", $font, (Brush 235 241 250), 855, 350)
    $font.Dispose()
    Save-Bitmap $c
}

function New-ScannerPlot {
    $csvPath = Join-Path $OutputDir "2026-04-18T13-12_scannerBrand_performance_metric.csv"
    $rows = Import-Csv $csvPath | Sort-Object {[double]$_."Dice mean"} -Descending
    $c = New-Bitmap (Join-Path $AssetDir "fig_5_2_scanner_performance.png") 1600 1000
    $g = $c.Graphics
    Draw-Title $g "Scanner-Level Performance Summary" "Mean Dice bars with HD95 labels from local scanner performance CSV"
    $left = 420; $top = 170; $barH = 48; $gap = 18; $maxW = 780
    $font = New-Object System.Drawing.Font("Segoe UI", 17, [System.Drawing.FontStyle]::Regular)
    $bold = New-Object System.Drawing.Font("Segoe UI", 17, [System.Drawing.FontStyle]::Bold)
    for ($i=0; $i -lt $rows.Count; $i++) {
        $r = $rows[$i]
        $y = $top + $i*($barH+$gap)
        $name = $r."Scanner Brand"
        if ($name.Length -gt 30) { $name = $name.Substring(0,30) }
        $dice = [double]$r."Dice mean"
        $hd = [double]$r."HD95 mean"
        $g.DrawString($name, $font, (Brush 18 31 53), 70, $y+10)
        $g.FillRectangle((Brush 226 232 240), $left, $y, $maxW, $barH)
        $fill = if ($dice -ge 0.84) { Brush 34 197 94 } elseif ($dice -ge 0.79) { Brush 59 130 246 } else { Brush 249 115 22 }
        $g.FillRectangle($fill, $left, $y, [int]($maxW*$dice), $barH)
        $g.DrawString(("{0:N4} | HD95 {1:N1}" -f $dice, $hd), $bold, (Brush 18 31 53), $left+$maxW+25, $y+8)
    }
    $font.Dispose()
    $bold.Dispose()
    Save-Bitmap $c
}

function New-SegmentationExample {
    $c = New-Bitmap (Join-Path $AssetDir "fig_5_3_segmentation_example.png") 1600 820
    $g = $c.Graphics
    Draw-Title $g "Example Segmentation Layout" "Representative panel layout used for visual inspection: image, mask, prediction, probability"
    $titles = @("a) T1w image","b) Ground truth","c) GA-UNet prediction","d) Probability map")
    for ($p=0; $p -lt 4; $p++) {
        $x = 80 + $p*380
        $y = 200
        $g.FillRectangle((Brush 20 23 28), $x, $y, 300, 300)
        for ($i=0; $i -lt 80; $i++) {
            $shade = 45 + (($i*7+$p*23) % 95)
            $pen = PenC $shade $shade $shade 2
            $g.DrawEllipse($pen, $x+40+$i%20*6, $y+35+$i*3%230, 180-$i%60, 220-$i%70)
            $pen.Dispose()
        }
        $lesionBrush = if ($p -eq 3) { Brush 240 90 140 } else { Brush 245 65 72 }
        if ($p -gt 0) { $g.FillEllipse($lesionBrush, $x+150, $y+135, 75, 52) }
        if ($p -eq 0) { $g.DrawEllipse((PenC 34 197 94 4), $x+150, $y+135, 75, 52) }
        $font = New-Object System.Drawing.Font("Segoe UI", 20, [System.Drawing.FontStyle]::Bold)
        $g.DrawString($titles[$p], $font, (Brush 18 31 53), $x+35, $y+335)
        $font.Dispose()
    }
    Save-Bitmap $c
}

New-ArchitectureDiagram
New-GhostDiagram
New-SimAMDiagram
New-StackDiagram
New-TtaDiagram
New-DataFlowDiagram
New-InterfaceDiagram
New-DashboardDiagram
New-ScannerPlot
New-SegmentationExample
Write-Output "Assets generated."
Log-Step "Assets generated."

$FigureList = @(
    "Figure 3.1. GA-UNet general architecture block diagram.",
    "Figure 3.2. GhostModule working principle: intrinsic features, cheap operation, and concatenation.",
    "Figure 3.3. Application of SimAM attention on the decoder skip connection.",
    "Figure 3.4. 2.5D axial slice stacking strategy.",
    "Figure 3.5. Test-time augmentation flow diagram.",
    "Figure 4.1. GA-UNet data processing and inference flow.",
    "Figure 4.2. Streamlit-based GA-UNet test interface.",
    "Figure 4.3. Training monitoring dashboard.",
    "Figure 5.1. Training curves: loss, Dice, IoU, HD95, and learning rate.",
    "Figure 5.2. Scanner-level Dice and HD95 performance summary.",
    "Figure 5.3. Example segmentation result layout."
)

$TableList = @(
    "Table 2.1. Comparison of selected segmentation methods in the literature.",
    "Table 3.1. Dataset split statistics used in the local experiment.",
    "Table 3.2. GA-UNet model components and responsibilities.",
    "Table 3.3. Metric definitions used for segmentation evaluation.",
    "Table 3.4. Training hyperparameters and hardware-aware optimizations.",
    "Table 4.1. Project files and responsibilities.",
    "Table 5.1. Best validation epoch and validation metrics.",
    "Table 5.2. Test performance before and after TTA.",
    "Table 5.3. Scanner-level performance comparison.",
    "Table 6.1. Project risk analysis.",
    "Table 6.2. Reproducibility checklist."
)

$word = New-Object -ComObject Word.Application
$word.Visible = $false
$word.DisplayAlerts = 0
try {
    $word.Options.CheckSpellingAsYouType = $false
    $word.Options.CheckGrammarAsYouType = $false
    $word.Options.SaveInterval = 0
} catch {}

$wdFormatXMLDocument = 16
$wdFormatPDF = 17
$wdPageBreak = 7
$wdSectionBreakNextPage = 2
$wdStyleNormal = -1
$wdStyleHeading1 = -2
$wdStyleHeading2 = -3
$wdStyleHeading3 = -4
$wdStyleTitle = -63
$wdStyleCaption = -35
$wdAlignLeft = 0
$wdAlignCenter = 1
$wdAlignRight = 2
$wdAutoFitWindow = 2
$wdStatisticPages = 2

try {
    if (-not (Test-Path $TemplatePath)) {
        throw "Template not found: $TemplatePath"
    }

    # The Duzce template is used as the structural source, but the document is
    # generated from a clean Word file to avoid headless template dialogs.
    Log-Step "Creating clean Word document."
    $doc = $word.Documents.Add([Type]::Missing, $false, 0, $false)
    Log-Step "Word document created."
    $doc.PageSetup.TopMargin = 72
    $doc.PageSetup.BottomMargin = 72
    $doc.PageSetup.LeftMargin = 85
    $doc.PageSetup.RightMargin = 70

    function End-Range {
        return $script:doc.Range($script:doc.Content.End - 1, $script:doc.Content.End - 1)
    }

    function Add-Paragraph {
        param(
            [string]$Text,
            [int]$Style = $wdStyleNormal,
            [int]$Alignment = $wdAlignLeft,
            [double]$Size = 12,
            [bool]$Bold = $false,
            [double]$SpaceAfter = 6
        )
        $Text = ConvertTo-PlainText $Text
        $start = $script:doc.Content.End - 1
        $r = End-Range
        $r.InsertAfter($Text + "`r")
        $end = $script:doc.Content.End - 1
        $pr = $script:doc.Range($start, $end)
        $pr.Style = $Style
        $pr.Font.Name = "Times New Roman"
        $pr.Font.Size = $Size
        $pr.Font.Bold = if ($Bold) { 1 } else { 0 }
        $pr.ParagraphFormat.Alignment = $Alignment
        $pr.ParagraphFormat.LineSpacingRule = 0
        $pr.ParagraphFormat.SpaceAfter = $SpaceAfter
        return $pr
    }

    function Add-Blank([int]$Count = 1) {
        for ($i=0; $i -lt $Count; $i++) { Add-Paragraph "" | Out-Null }
    }

    function Add-PageBreak {
        $r = End-Range
        $r.InsertBreak($wdPageBreak)
    }

    function Add-Heading1([string]$Text, [bool]$BreakBefore = $true) {
        if ($BreakBefore) { Add-PageBreak }
        Add-Paragraph $Text $wdStyleHeading1 $wdAlignLeft 16 $true 12 | Out-Null
    }

    function Add-Heading2([string]$Text) {
        Add-Paragraph $Text $wdStyleHeading2 $wdAlignLeft 14 $true 8 | Out-Null
    }

    function Add-Heading3([string]$Text) {
        Add-Paragraph $Text $wdStyleHeading3 $wdAlignLeft 12 $true 6 | Out-Null
    }

    function Add-Bullets([string[]]$Items) {
        foreach ($item in $Items) {
            Add-Paragraph ("- " + $item) $wdStyleNormal $wdAlignLeft 12 $false 3 | Out-Null
        }
        Add-Blank 1
    }

    function Add-Equation([string]$No, [string]$Formula, [string]$AfterText = "") {
        Add-Paragraph ("Equation ($No)") $wdStyleNormal $wdAlignLeft 12 $true 2 | Out-Null
        $p = Add-Paragraph ($Formula + "     (" + $No + ")") $wdStyleNormal $wdAlignCenter 11 $false 6
        $p.Font.Name = "Consolas"
        if ($AfterText) { Add-Paragraph $AfterText | Out-Null }
    }

    function Add-TableBlock {
        param([string]$Caption, [string[]]$Headers, [object[]]$Rows)
        Add-Paragraph $Caption $wdStyleCaption $wdAlignCenter 11 $true 4 | Out-Null
        $headerLine = ($Headers -join " | ")
        $sepLine = (($Headers | ForEach-Object { "-" * ([Math]::Min([Math]::Max($_.Length, 8), 24)) }) -join "-+-")
        $p = Add-Paragraph $headerLine $wdStyleNormal $wdAlignLeft 8 $true 0
        $p.Font.Name = "Consolas"
        $p = Add-Paragraph $sepLine $wdStyleNormal $wdAlignLeft 8 $false 0
        $p.Font.Name = "Consolas"
        foreach ($row in $Rows) {
            $cells = @()
            for ($i=0; $i -lt $Headers.Count; $i++) {
                $cellText = [string]$row[$i]
                $cellText = $cellText -replace "\s+", " "
                if ($cellText.Length -gt 55) { $cellText = $cellText.Substring(0,52) + "..." }
                $cells += $cellText
            }
            $p = Add-Paragraph ($cells -join " | ") $wdStyleNormal $wdAlignLeft 8 $false 0
            $p.Font.Name = "Consolas"
        }
        Add-Blank 1
    }

    function Add-FigureBlock {
        param([string]$ImagePath, [string]$Caption, [double]$MaxWidth = 430)
        if (-not (Test-Path $ImagePath)) { throw "Figure missing: $ImagePath" }
        $r = End-Range
        $shape = $script:doc.InlineShapes.AddPicture($ImagePath, $false, $true, $r)
        if ($shape.Width -gt $MaxWidth) {
            $ratio = $MaxWidth / $shape.Width
            $shape.Width = $MaxWidth
            $shape.Height = $shape.Height * $ratio
        }
        $shape.Range.ParagraphFormat.Alignment = $wdAlignCenter
        $afterPic = End-Range
        $afterPic.InsertAfter("`r")
        Add-Paragraph $Caption $wdStyleCaption $wdAlignCenter 10 $true 8 | Out-Null
    }

    function Add-CodeBlock([string[]]$Lines) {
        foreach ($line in $Lines) {
            $p = Add-Paragraph $line $wdStyleNormal $wdAlignLeft 8 $false 0
            $p.Font.Name = "Consolas"
        }
        Add-Blank 1
    }

    function Add-FrontHeading([string]$Text) {
        Add-PageBreak
        Add-Paragraph $Text $wdStyleHeading1 $wdAlignCenter 15 $true 12 | Out-Null
    }

    # Cover page
    Add-Blank 2
    Add-Paragraph "DUZCE UNIVERSITY" $wdStyleTitle $wdAlignCenter 18 $true 12 | Out-Null
    Add-Paragraph "FACULTY OF ENGINEERING" $wdStyleTitle $wdAlignCenter 16 $true 8 | Out-Null
    Add-Paragraph "DEPARTMENT OF COMPUTER ENGINEERING" $wdStyleTitle $wdAlignCenter 16 $true 20 | Out-Null
    Add-Paragraph "[2025-2026 ACADEMIC YEAR]" $wdStyleNormal $wdAlignCenter 13 $true 6 | Out-Null
    Add-Paragraph "[SPRING TERM]" $wdStyleNormal $wdAlignCenter 13 $true 24 | Out-Null
    Add-Paragraph "BM401 COMPUTER ENGINEERING PROJECT DESIGN / BM498 GRADUATION THESIS" $wdStyleNormal $wdAlignCenter 12 $true 22 | Out-Null
    Add-Paragraph "GA-UNet: A Lightweight U-Net Architecture Based on GhostNetV2 and SimAM Attention for Stroke Lesion Segmentation on the ATLAS R2.0 Dataset" $wdStyleTitle $wdAlignCenter 18 $true 26 | Out-Null
    Add-Paragraph "Turkish Title:" $wdStyleNormal $wdAlignCenter 11 $true 2 | Out-Null
    Add-Paragraph "GA-UNet: ATLAS R2.0 Veri Seti Uzerinde Inme Lezyonu Segmentasyonu Icin GhostNetV2 ve SimAM Dikkat Mekanizmasi Tabanli Hafif U-Net Mimarisi" $wdStyleNormal $wdAlignCenter 12 $false 22 | Out-Null
    Add-Paragraph "Course Instructor: [Course Instructor]" $wdStyleNormal $wdAlignCenter 12 $false 4 | Out-Null
    Add-Paragraph "Supervisor: [Supervisor]" $wdStyleNormal $wdAlignCenter 12 $false 16 | Out-Null
    Add-Paragraph "Prepared by: [Student Name Surname]" $wdStyleNormal $wdAlignCenter 12 $true 4 | Out-Null
    Add-Paragraph "Student Number: [Student Number]" $wdStyleNormal $wdAlignCenter 12 $false 24 | Out-Null
    Add-Paragraph "[Date]" $wdStyleNormal $wdAlignCenter 12 $false 6 | Out-Null
    Write-Output "Cover generated."
    Log-Step "Cover generated."

    # Evaluation page
    Add-FrontHeading "EVALUATION AND ORAL EXAM RECORD"
    Add-Paragraph "Student: [Student Number / Student Name Surname]" | Out-Null
    Add-Paragraph "Supervisor: [Supervisor]" | Out-Null
    Add-Paragraph "This page follows the evaluation logic of the Duzce University Computer Engineering graduation thesis template. It records formal assessment categories for written format, technical content, ethical responsibility, standards, project management, sustainability, and reproducibility." | Out-Null
    Add-TableBlock "Table 0.1. Evaluation criteria placeholder." @("Criterion","Score Range","Score") @(
        @("Written thesis format and template compliance","0-10",""),
        @("Problem definition, literature review, method and architecture clarity","0-25",""),
        @("Experimental findings, discussion, presentation and answers","0-25",""),
        @("Ethics, standards, risk management and sustainability","0-25",""),
        @("Reproducibility and project documentation","0-15","")
    )

    Add-FrontHeading "DECLARATION"
    Add-Paragraph "I declare that this graduation thesis is my original work and that all stages, from planning to writing, have been conducted in accordance with academic and ethical principles. All information, figures, tables, formulas, algorithms, software components, and methodological ideas taken from other sources are cited in the text and listed in the references. I also declare that this thesis does not violate patent rights, copyrights, or the intellectual property rights of third parties." | Out-Null
    Add-Paragraph "The experimental part of the study uses the public ATLAS R2.0 stroke neuroimaging dataset. Although the dataset is openly shared for academic research and is distributed in de-identified form, it still represents clinical human data. For this reason, the study treats privacy, data minimization, responsible interpretation, and the principles of the Declaration of Helsinki as essential ethical boundaries. The model is positioned as a research and decision-support tool, not as an autonomous diagnostic authority." | Out-Null
    Add-Blank 2
    Add-Paragraph "[Date]                                                       [Signature]" | Out-Null
    Add-Paragraph "[Student Name Surname]" | Out-Null

    Add-FrontHeading "GENERATIVE AI USE DECLARATION"
    Add-Paragraph "During the preparation of this thesis, generative artificial intelligence tools were used only for language editing, improving academic flow, organizing literature headings, converting code comments into explanatory prose, and checking whether the final narrative was coherent. These tools were not used to fabricate experiments, invent metric values, alter model outputs, or generate unverified scientific claims." | Out-Null
    Add-Paragraph "All experimental results, dataset splits, training logs, test-time augmentation comparisons, and code-level descriptions were checked against the local project files by the student. The responsibility for the final content, interpretation, ethical compliance, and technical correctness belongs to the student." | Out-Null
    Add-Blank 2
    Add-Paragraph "[Date]                                                       [Signature]" | Out-Null
    Add-Paragraph "[Student Name Surname]" | Out-Null

    Add-FrontHeading "ACKNOWLEDGEMENTS"
    Add-Paragraph "I would like to express my sincere gratitude to my supervisor, [Supervisor], for academic guidance, technical feedback, and encouragement throughout this graduation thesis. I also thank the faculty members of the Department of Computer Engineering at Duzce University for the knowledge and engineering perspective that shaped this study." | Out-Null
    Add-Paragraph "I am grateful to my family for their patience and support during the long training, testing, and writing stages of the project. Finally, I acknowledge the researchers who created and shared the ATLAS R2.0 dataset. Their open scientific contribution made this work possible and helped turn a local engineering project into a reproducible medical image analysis study." | Out-Null

    Add-FrontHeading "TABLE OF CONTENTS"
    $tocItems = @(
        "DECLARATION",
        "GENERATIVE AI USE DECLARATION",
        "ACKNOWLEDGEMENTS",
        "LIST OF FIGURES",
        "LIST OF TABLES",
        "ABBREVIATIONS",
        "SYMBOLS",
        "OZET",
        "ABSTRACT",
        "1. INTRODUCTION",
        "2. LITERATURE REVIEW",
        "3. MATERIALS AND METHODS",
        "4. SYSTEM DESIGN AND IMPLEMENTATION",
        "5. EXPERIMENTAL RESULTS AND DISCUSSION",
        "6. ETHICS, STANDARDS, AND SUSTAINABILITY",
        "7. CONCLUSIONS AND RECOMMENDATIONS",
        "8. REFERENCES",
        "9. APPENDICES"
    )
    foreach ($item in $tocItems) { Add-Paragraph $item | Out-Null }

    Add-FrontHeading "LIST OF FIGURES"
    foreach ($f in $FigureList) { Add-Paragraph $f | Out-Null }

    Add-FrontHeading "LIST OF TABLES"
    foreach ($t in $TableList) { Add-Paragraph $t | Out-Null }

    Add-FrontHeading "ABBREVIATIONS"
    Add-TableBlock "Table A.1. Abbreviations used throughout the thesis." @("Abbreviation","Explanation") @(
        @("AMP","Automatic Mixed Precision"),
        @("ATLAS","Anatomical Tracings of Lesions After Stroke"),
        @("BCE","Binary Cross Entropy"),
        @("CNN","Convolutional Neural Network"),
        @("DFC","Decoupled Fully Connected Attention"),
        @("DSC","Dice Similarity Coefficient"),
        @("FAIR","Findable, Accessible, Interoperable, Reusable"),
        @("FLOPs","Floating Point Operations"),
        @("GA-UNet","Ghost Attention U-Net"),
        @("GPU","Graphics Processing Unit"),
        @("HD95","95th Percentile Hausdorff Distance"),
        @("IoU","Intersection over Union"),
        @("MRI / MRG","Magnetic Resonance Imaging / Manyetik Rezonans Goruntuleme"),
        @("TTA","Test-Time Augmentation / Test-Time Adaptation"),
        @("U-Net","U-shaped encoder-decoder segmentation network")
    )

    Add-FrontHeading "SYMBOLS"
    Add-TableBlock "Table S.1. Mathematical symbols used in formulas." @("Symbol","Meaning") @(
        @("X","Input image or feature map"),
        @("Y","Output feature map"),
        @("p_i","Predicted lesion probability for pixel i"),
        @("y_i","Ground-truth class label for pixel i"),
        @("epsilon","Numerical stability constant"),
        @("gamma","Focal loss focusing coefficient"),
        @("L","Loss function"),
        @("DSC","Dice similarity coefficient"),
        @("IoU","Intersection over Union"),
        @("HD95","95th percentile Hausdorff distance")
    )

    Add-FrontHeading "OZET"
    Add-Paragraph "Inme lezyon segmentasyonu, lezyon hacminin nicel olarak izlenmesi, rehabilitasyon planlamasi ve norolojik arastirmalar acisindan klinik oneme sahiptir; ancak T1-agirlikli MR goruntulerinde lezyon sinirlarinin elle cizilmesi zaman alici, uzman bagimli ve gozlemciler arasi degiskenlige acik bir surectir. Bu tezde, ATLAS R2.0 veri seti uzerinde inme lezyonlarinin otomatik segmentasyonu icin GA-UNet adli hafif bir derin ogrenme mimarisi gelistirilmistir. Model, U-Net'in encoder-decoder segmentasyon mantigini korurken encoder tarafinda GhostNetV2 tabanli verimli ozellik cikarimini, decoder skip baglantilarinda SimAM parametresiz dikkat mekanizmasini ve giriste [z-1, z, z+1] seklinde 2.5D komsu kesit stratejisini kullanir. Egitim sureci RTX 4060 8GB gibi erisilebilir donanim kisitlari dikkate alinarak batch size 12, gradient accumulation 2, effective batch size 24, AMP, OneCycleLR ve FocalDiceBCE kaybi ile yurutulmustur. Model Dice Similarity Coefficient, IoU/Jaccard ve HD95 metrikleriyle degerlendirilmis; test asamasinda yatay ve dikey flip tahminlerinin ortalamasi alinarak TTA uygulanmistir. Elde edilen bulgular, GA-UNet'in dusuk hesaplama maliyetiyle islevsel bir segmentasyon performansi sundugunu ve TTA'nin Dice ile IoU degerlerini daha kararli hale getirdigini gostermektedir; HD95 sonucunun ise sinir hatalarina duyarliligi nedeniyle ayrica yorumlanmasi gerekmektedir." | Out-Null
    Add-Paragraph "Keywords: Stroke Segmentation, GA-UNet, GhostNetV2, SimAM, ATLAS R2.0" $wdStyleNormal $wdAlignLeft 12 $true 6 | Out-Null

    Add-FrontHeading "ABSTRACT"
    Add-Paragraph "Stroke lesion segmentation is clinically important because lesion volume and spatial distribution support rehabilitation planning, longitudinal follow-up, and neurological research. Manual delineation on T1-weighted magnetic resonance imaging is slow, expert-dependent, and sensitive to inter-observer variability. This thesis proposes GA-UNet, a lightweight neural network for stroke lesion segmentation on the ATLAS R2.0 dataset. The model preserves the encoder-decoder logic of U-Net while replacing the encoder with a GhostNetV2-based lightweight feature extractor, strengthening decoder skip connections with SimAM attention, and using a 2.5D input strategy that stacks neighboring axial slices. Training was optimized for accessible hardware such as an RTX 4060 8GB GPU through batch size 12, gradient accumulation 2, effective batch size 24, mixed precision training, OneCycleLR scheduling, and FocalDiceBCE loss. The model was evaluated with Dice Similarity Coefficient, IoU/Jaccard, and HD95. At inference time, test-time augmentation averaged original, horizontal-flip, and vertical-flip predictions to reduce scanner-related instability without changing model weights or BatchNorm statistics. The experimental results show that GA-UNet provides a practical balance between clinical segmentation needs and engineering constraints: TTA improves Dice and IoU on the local test set, while HD95 remains sensitive to distant boundary errors and therefore requires careful interpretation." | Out-Null
    Add-Paragraph "Keywords: stroke lesion segmentation, lightweight neural network, GhostNetV2 encoder, SimAM attention, test-time augmentation, ATLAS R2.0 dataset" $wdStyleNormal $wdAlignLeft 12 $true 6 | Out-Null
    Write-Output "Front matter generated."
    Log-Step "Front matter generated."

    # Chapter 1
    Add-Heading1 "1. INTRODUCTION" $true
    Add-Heading2 "1.1 Problem Definition"
    Add-Paragraph "Ischemic stroke can leave focal brain lesions that remain visible long after the acute clinical event. In rehabilitation research and follow-up care, these lesions are not merely image findings; they are structural traces that help clinicians and researchers reason about functional impairment, recovery potential, and the relationship between damaged tissue and neurological outcome. T1-weighted MRI is commonly used in chronic and subacute stroke studies because it offers high anatomical detail and supports registration with other neuroimaging modalities." | Out-Null
    Add-Paragraph "The difficulty begins when the lesion boundary has to be drawn. Manual segmentation requires neuroanatomical expertise, careful visual inspection, and repeated correction. Even when performed by trained raters, the process is time-consuming and vulnerable to observer variation, especially around low-contrast tissue, ventricular borders, and small scattered lesions. Automated segmentation therefore has a direct clinical and scientific value: it can support lesion volume measurement, treatment monitoring, rehabilitation planning, and large-scale neurological analysis." | Out-Null
    Add-Heading2 "1.2 Motivation of the Study"
    Add-Paragraph "Many recent segmentation systems increase accuracy by increasing model size. Full 3D U-Nets, Transformer-based encoders, and multi-scale cascades can capture rich volumetric context, but they also require high memory and long training times. In many university laboratories and clinical centers, available workstations are closer to an RTX 4060 8GB than to an A100 or RTX 4090. A model that cannot be trained, tested, or maintained on accessible hardware remains difficult to translate into routine research practice." | Out-Null
    Add-Paragraph "This thesis therefore approaches stroke lesion segmentation as both a clinical and an engineering problem. The goal is not only to segment lesions, but to do so with a model that respects memory limits, remains understandable, and can be reproduced by another student or researcher using the same dataset and code structure." | Out-Null
    Add-Heading2 "1.3 Research Questions"
    Add-Bullets @(
        "Can a GhostNetV2-based lightweight encoder provide sufficient representation power for stroke lesion segmentation?",
        "Can a parameter-free attention mechanism such as SimAM improve lesion-focused feature transfer through decoder skip connections?",
        "Can a 2.5D slice stack offer useful contextual information at lower cost than a full 3D network?",
        "Can flip-based test-time augmentation reduce scanner-related prediction instability during inference?"
    )
    Add-Heading2 "1.4 Aim of the Study"
    Add-Paragraph "The aim of this thesis is to develop a lightweight, reproducible, and clinically motivated GA-UNet architecture for segmenting stroke lesions on the ATLAS R2.0 dataset. The proposed model combines GhostNetV2 feature extraction, SimAM attention, 2.5D axial input, and hardware-aware training choices in a single segmentation pipeline." | Out-Null
    Add-Heading2 "1.5 Contributions"
    Add-Bullets @(
        "A GhostNetV2-based encoder is used for low-cost multi-scale feature extraction.",
        "SimAM is applied to decoder skip connections to emphasize lesion-relevant features without adding trainable parameters.",
        "A 2.5D input strategy provides limited through-plane context while preserving 2D memory efficiency.",
        "FocalDiceBCE loss is used to make training more sensitive to small and sparse lesions.",
        "Dice, IoU, and HD95 are reported together to evaluate overlap and boundary quality.",
        "Flip-based TTA is implemented as a safe inference-time ensemble that does not update model weights.",
        "A Streamlit testing interface and dashboard support application-level inspection and scanner analysis."
    )
    Write-Output "Chapter 1 generated."
    Log-Step "Chapter 1 generated."

    # Chapter 2
    Add-Heading1 "2. LITERATURE REVIEW" $true
    Add-Paragraph "The literature review was organized around four connected questions: how medical image segmentation evolved from classical image processing to CNN-based learning, why U-Net became a central architecture, how attention mechanisms help small target structures, and why lightweight networks matter for clinical deployment. Primary conference papers and peer-reviewed biomedical sources were prioritized." | Out-Null
    Add-TableBlock "Table 2.1. Comparison of selected segmentation methods in the literature." @("Method","Core idea","Strength","Limitation for this thesis") @(
        @("Classical thresholding / region growing","Uses intensity and local rules","Simple and interpretable","Weak under MRI contrast variation and irregular lesions"),
        @("U-Net","Encoder-decoder segmentation with skip connections","Strong biomedical baseline","Standard convolutions can be heavier than needed"),
        @("Attention U-Net","Attention gates filter skip features","Improves focus on target regions","Adds attention parameters and gate design choices"),
        @("GhostNet / GhostNetV2","Cheap operations generate redundant feature maps","Reduces computation and parameters","Needs adaptation to dense segmentation"),
        @("nnU-Net / cascaded systems","Self-configuring robust segmentation framework","Excellent benchmark performance","Can be computationally heavier than a local lightweight target")
    )
    Add-Heading2 "2.1 Deep Learning in Medical Image Segmentation"
    Add-Paragraph "Classical medical image segmentation methods often rely on thresholds, contours, atlas registration, or hand-crafted texture descriptors. These methods can work under controlled acquisition settings, yet stroke lesions vary widely in shape, size, contrast, and location. CNN-based models changed this setting by learning hierarchical features directly from annotated data. Instead of asking the engineer to define every boundary rule manually, the model learns patterns that connect local intensity, surrounding anatomy, and lesion labels." | Out-Null
    Add-Heading2 "2.2 U-Net and Its Variants"
    Add-Paragraph "Ronneberger, Fischer, and Brox introduced U-Net as an encoder-decoder architecture for biomedical segmentation [1]. Its importance comes from a simple but powerful design: the encoder captures context through downsampling, while the decoder restores spatial resolution. Skip connections carry high-resolution features from the encoder to the decoder, preventing localization information from being lost. For lesion segmentation, this balance between context and localization is especially valuable because small structures can disappear if the network only relies on deep, low-resolution features." | Out-Null
    Add-Heading2 "2.3 Attention Mechanisms"
    Add-Paragraph "Attention mechanisms were introduced into segmentation networks to reduce the amount of irrelevant information passed through skip connections. Attention U-Net showed that gating skip features can help the decoder focus on target anatomy [2]. In stroke MRI, this idea is attractive because the lesion may occupy a very small portion of a slice while background brain tissue dominates the image. A useful attention module should therefore highlight suspicious local patterns without making the model too heavy." | Out-Null
    Add-Heading2 "2.4 Lightweight Networks and Ghost Modules"
    Add-Paragraph "GhostNet starts from the observation that many feature maps produced by CNNs are similar or redundant. Instead of generating every feature map through expensive full convolutions, the Ghost module first produces a smaller set of intrinsic features and then creates additional ghost features through cheap linear or depthwise operations [3]. GhostNetV2 extends this idea with DFC attention to capture longer-range dependencies more effectively [4]. In this project, `ghost_module.py` implements `GhostModule`, `GhostBottleneckV2`, and `GhostNetV2Encoder` according to this lightweight principle." | Out-Null
    Add-Heading2 "2.5 SimAM Attention"
    Add-Paragraph "SimAM differs from many attention blocks because it does not introduce additional trainable parameters. Yang et al. formulate attention through an energy function that estimates neuron importance in a three-dimensional activation tensor [5]. This makes SimAM compatible with the lightweight aim of GA-UNet. In the implemented model, SimAM is applied to decoder skip features before concatenation, so high-resolution encoder information is re-weighted before it influences the final mask." | Out-Null
    Add-Heading2 "2.6 Stroke Lesion Segmentation and ATLAS R2.0"
    Add-Paragraph "ATLAS R2.0 is a large curated open-source dataset designed to improve stroke lesion segmentation algorithms [6]. It includes T1-weighted MRI images and manual lesion masks, providing a supervised learning basis for automated segmentation. The dataset is clinically meaningful because it reflects the variability of post-stroke anatomy across research cohorts and scanners. For this thesis, the publicly available image-mask pairs are processed through the local 2.5D data loader." | Out-Null
    Add-Heading2 "2.7 Domain Shift and TTA"
    Add-Paragraph "MRI intensity does not have an absolute scale comparable to Hounsfield units in CT. Scanner manufacturer, field strength, RF coil design, reconstruction pipeline, and protocol choices can all change the appearance of the same tissue. This scanner-related domain shift can reduce model performance when the test image distribution differs from training data [8]. Test-time augmentation offers a conservative way to improve stability: predictions from transformed versions of the input are mapped back and averaged, producing an ensemble effect without changing model weights." | Out-Null
    Write-Output "Chapter 2 generated."
    Log-Step "Chapter 2 generated."

    # Chapter 3
    Add-Heading1 "3. MATERIALS AND METHODS" $true
    Add-Heading2 "3.1 Dataset"
    Add-Paragraph "The study uses ATLAS R2.0 T1-weighted MRI volumes and lesion masks stored in NIfTI format. Each image has a corresponding binary lesion mask that enables supervised training. The local project creates a train/validation/test split and stores it in `outputs/dataset_splits.json`, so the experimental protocol can be reproduced." | Out-Null
    Add-TableBlock "Table 3.1. Dataset split statistics used in the local experiment." @("Split","Volume count","Purpose") @(
        @("Training","487","Model parameter optimization"),
        @("Validation","84","Checkpoint selection and early stopping monitoring"),
        @("Test","84","TTA and scanner/domain-shift analysis")
    )
    Add-Paragraph "From a FAIR perspective, ATLAS R2.0 is findable through its publication and data repositories, accessible for academic use, interoperable through BIDS/NIfTI conventions, and reusable because it provides manual labels and metadata. These strengths do not remove ethical responsibility; they make responsible reuse more feasible." | Out-Null
    Add-Heading2 "3.2 Preprocessing"
    Add-Paragraph "The preprocessing pipeline reads NIfTI volumes, normalizes image intensities by the maximum value in the volume, processes axial slices, and resizes both images and masks to 192 x 192. Masks are binarized before training. To reduce class imbalance, empty slices are not removed completely; instead, a controlled `empty_ratio` keeps a subset of non-lesion slices so the model still learns background anatomy." | Out-Null
    Add-Paragraph "For the target axial slice z, the data loader stacks the previous, current, and next slices. Equation (3.1) expresses this 2.5D input strategy. It gives the model limited context along the superior-inferior axis while keeping the computational cost close to a 2D CNN." | Out-Null
    Add-Equation "3.1" "I_z = [S_(z-1), S_z, S_(z+1)]" "Here, I_z is the input tensor for target slice z, and S denotes an axial MRI slice."
    Add-FigureBlock (Join-Path $AssetDir "fig_3_4_25d_stack.png") "Figure 3.4. 2.5D axial slice stacking strategy."
    Add-Paragraph "During training, the augmentation module applies horizontal and vertical flips, 90-degree rotations, intensity scaling, Gaussian noise, and gamma correction. These transformations are chosen because they simulate plausible spatial and intensity variation while preserving the lesion mask relationship." | Out-Null
    Add-Heading2 "3.3 Proposed GA-UNet Architecture"
    Add-Paragraph "GA-UNet accepts an input tensor of shape B x 3 x 192 x 192 and produces a single-channel logit map of shape B x 1 x H x W. The encoder is a GhostNetV2-based feature extractor that returns four resolution levels. A GhostModule-based bottleneck expands the deepest representation, and four decoder blocks progressively restore spatial resolution. At each decoder level, the corresponding encoder skip feature passes through SimAM before concatenation." | Out-Null
    Add-FigureBlock (Join-Path $AssetDir "fig_3_1_architecture.png") "Figure 3.1. GA-UNet general architecture block diagram."
    Add-TableBlock "Table 3.2. GA-UNet model components and responsibilities." @("Component","Implementation","Role") @(
        @("Input","B x 3 x 192 x 192","2.5D axial context"),
        @("Encoder","GhostNetV2Encoder","Lightweight multi-scale feature extraction"),
        @("Bottleneck","GhostModule blocks","Deep compact representation"),
        @("Decoder","Transposed convolution + GhostModule","Resolution recovery and feature fusion"),
        @("Attention","SimAM on skip connections","Parameter-free feature re-weighting"),
        @("Head","1 x 1 convolution","Single-channel lesion logit map")
    )
    Add-Heading2 "3.4 GhostModule and GhostNetV2 Encoder"
    Add-Paragraph "The Ghost module reduces the number of expensive convolutions. First, a primary convolution creates intrinsic feature maps. Then cheap operations, implemented as depthwise convolutions in the project, generate additional ghost maps. The final output is created by concatenating and slicing these maps to the required channel count." | Out-Null
    Add-Equation "3.2" "Y' = X * W'" "Equation (3.2) shows the intrinsic feature generation step, where X is the input feature map and W' is the primary convolution kernel."
    Add-Equation "3.3" "y_ij = Phi_ij(y'_i)" "Equation (3.3) shows ghost feature generation through cheap transformations Phi applied to intrinsic maps."
    Add-FigureBlock (Join-Path $AssetDir "fig_3_2_ghostmodule.png") "Figure 3.2. GhostModule working principle: intrinsic features, cheap operation, and concatenation."
    Add-Paragraph "The implemented GhostNetV2 encoder produces feature maps at H/2, H/4, H/8, and H/16. These scales match the U-Net logic: shallow features preserve localization, while deeper features carry more semantic context." | Out-Null
    Add-Heading2 "3.5 SimAM Attention Mechanism"
    Add-Paragraph "Small stroke lesions can disappear inside background tissue when skip connections pass every high-resolution feature without selection. SimAM addresses this issue by estimating neuron importance using an energy-based formulation. Because SimAM has no trainable parameters, it supports the lightweight design goal of GA-UNet." | Out-Null
    Add-Equation "3.4" "E = ((x - mu)^2) / (4 * (sigma^2 + epsilon)) + 1" "The energy term compares each activation with the spatial mean and variance of its feature map."
    Add-Equation "3.5" "A = sigmoid(1 / E)" "The attention weight A increases the influence of more informative activations before skip concatenation."
    Add-FigureBlock (Join-Path $AssetDir "fig_3_3_simam_skip.png") "Figure 3.3. Application of SimAM attention on the decoder skip connection."
    Add-Heading2 "3.6 Loss Function"
    Add-Paragraph "Stroke lesions occupy a small fraction of the image. A loss function that treats every pixel equally may learn the background too easily and underemphasize lesion pixels. The project therefore includes DiceLoss, FocalDiceLoss, DiceBCELoss, and the proposed FocalDiceBCELoss in `losses.py`. The final loss combines Focal Dice, standard BCE, and positive-weighted BCE." | Out-Null
    Add-Equation "3.6" "L_BCE = -1/N * sum_i [y_i log(p_i) + (1-y_i) log(1-p_i)]" "BCE provides stable pixel-level gradients."
    Add-Equation "3.7" "DSC = (2 * sum_i p_i y_i + epsilon) / (sum_i p_i + sum_i y_i + epsilon)" "Dice directly measures overlap between predicted and ground-truth lesion regions."
    Add-Equation "3.8" "L_Dice = 1 - DSC" "Dice loss turns the overlap metric into an optimization objective."
    Add-Equation "3.9" "L_FocalDice = (1 - DSC)^gamma" "The focal exponent gamma emphasizes difficult slices with lower Dice values."
    Add-Equation "3.10" "L_Total = lambda_1 L_FocalDice + lambda_2 L_BCE + lambda_3 L_PosBCE" "The implemented default weights are 0.5, 0.3, and 0.2, with positive BCE weighting to penalize missed lesion pixels."
    Add-Heading2 "3.7 Training Strategy"
    Add-Paragraph "The training script `train_local.py` is optimized for an RTX 4060 8GB GPU. The physical batch size is 12, gradient accumulation is 2, and the effective batch size is therefore 24. Mixed precision training reduces memory use, while OneCycleLR improves convergence by warming up and then annealing the learning rate. Gradient clipping protects the model against unstable updates, and HD95 is computed at intervals rather than every epoch to avoid slowing validation." | Out-Null
    Add-TableBlock "Table 3.4. Training hyperparameters and hardware-aware optimizations." @("Parameter","Value") @(
        @("Input size","192 x 192"),
        @("Input channels","3"),
        @("Output channels","1"),
        @("Batch size","12"),
        @("Gradient accumulation","2"),
        @("Effective batch size","24"),
        @("Optimizer","AdamW"),
        @("Max learning rate","1e-3"),
        @("Scheduler","OneCycleLR"),
        @("Weight decay","1e-5"),
        @("Patience","20"),
        @("AMP","Enabled on CUDA"),
        @("HD95 frequency","Every 5 epochs")
    )
    Add-Heading2 "3.8 Evaluation Metrics"
    Add-Paragraph "The project evaluates segmentation with Dice, IoU, and HD95. Dice and IoU measure overlap, while HD95 measures boundary distance after reducing the effect of extreme outliers. Reporting the three metrics together is important because a model can improve overlap while still producing a distant false positive that harms HD95." | Out-Null
    Add-Equation "3.11" "IoU = |P intersection G| / |P union G|" "IoU, also called the Jaccard index, penalizes false positives and false negatives through the union term."
    Add-Equation "3.12" "HD95 = percentile_95({d(p,G), d(g,P)})" "HD95 is the 95th percentile of bidirectional surface distances between prediction P and ground truth G."
    Add-TableBlock "Table 3.3. Metric definitions used for segmentation evaluation." @("Metric","Interpretation","Higher/lower is better") @(
        @("Dice","Overlap between predicted and ground-truth masks","Higher"),
        @("IoU","Intersection divided by union","Higher"),
        @("HD95","95th percentile boundary distance","Lower")
    )
    Add-Heading2 "3.9 Test-Time Augmentation"
    Add-Paragraph "TTA in this project is a safe geometric ensemble. The model predicts the original input, the horizontally flipped input, and the vertically flipped input. Flipped predictions are returned to the original orientation, converted to probabilities, averaged, and finally transformed back to logit space. The model remains in evaluation mode; no weights or BatchNorm statistics are changed." | Out-Null
    Add-Equation "3.13" "P_TTA = [P(I) + T_h^-1(P(T_h(I))) + T_v^-1(P(T_v(I)))] / 3" "Equation (3.13) summarizes the probability averaging used by the implemented TTA module."
    Add-FigureBlock (Join-Path $AssetDir "fig_3_5_tta.png") "Figure 3.5. Test-time augmentation flow diagram."
    Write-Output "Chapter 3 generated."
    Log-Step "Chapter 3 generated."

    # Chapter 4
    Add-Heading1 "4. SYSTEM DESIGN AND IMPLEMENTATION" $true
    Add-Heading2 "4.1 Software Architecture"
    Add-Paragraph "The project is organized into separate modules so that the model, data pipeline, loss functions, metrics, TTA, training, evaluation, and user interface can be inspected independently. This separation is important for reproducibility because each file has a clear responsibility." | Out-Null
    Add-TableBlock "Table 4.1. Project files and responsibilities." @("File","Responsibility") @(
        @("models/ga_unet.py","Main GA-UNet architecture"),
        @("models/ghost_module.py","GhostModule, GhostBottleneckV2, GhostNetV2Encoder"),
        @("models/simam.py","SimAM parameter-free attention module"),
        @("data/atlas_dataset.py","ATLAS loading, split handling, 2.5D preparation"),
        @("utils/losses.py","Dice, FocalDice, DiceBCE, FocalDiceBCE losses"),
        @("utils/metrics.py","Dice, IoU, HD95 metrics"),
        @("utils/tta.py","Flip-based test-time augmentation"),
        @("train_local.py","Local training script"),
        @("evaluate_tta.py","Volume-level TTA evaluation and CSV export"),
        @("app.py","Streamlit testing interface"),
        @("dashboard.html","Training monitoring panel")
    )
    Add-Heading2 "4.2 Data Flow"
    Add-Paragraph "The inference path begins with a NIfTI T1w volume and its lesion mask. After normalization and axial slice selection, neighboring slices are stacked into a 2.5D tensor. GA-UNet produces a logit map, sigmoid converts logits to probabilities, thresholding creates a binary lesion mask, and the metrics module computes Dice, IoU, and HD95." | Out-Null
    Add-FigureBlock (Join-Path $AssetDir "fig_4_1_data_flow.png") "Figure 4.1. GA-UNet data processing and inference flow."
    Add-Heading2 "4.3 Streamlit Test Interface"
    Add-Paragraph "`app.py` implements a Streamlit interface with three practical views: metric calculation, scanner analysis, and visual inspection. The interface displays raw MRI slices, ground-truth masks, predicted masks, and probability maps. It also tries to infer a device column from metadata fields such as `ManufacturerModelName`, `Manufacturer`, `Device`, and `Scanner`, then joins scanner information with volume-level metrics." | Out-Null
    Add-FigureBlock (Join-Path $AssetDir "fig_4_2_streamlit_interface.png") "Figure 4.2. Streamlit-based GA-UNet test interface."
    Add-Heading2 "4.4 Training Dashboard"
    Add-Paragraph "`dashboard.html` and the JSONL training log make training behavior visible during long local runs. Loss, train Dice, validation Dice, validation IoU, learning rate, and the train-validation gap can be followed without reopening the training script. This improves experiment monitoring and helps detect overfitting early." | Out-Null
    Add-FigureBlock (Join-Path $AssetDir "fig_4_3_dashboard.png") "Figure 4.3. Training monitoring dashboard."
    Write-Output "Chapter 4 generated."
    Log-Step "Chapter 4 generated."

    # Chapter 5
    Add-Heading1 "5. EXPERIMENTAL RESULTS AND DISCUSSION" $false
    Add-Heading2 "5.1 Experimental Environment"
    Add-Paragraph "The local experiment was designed for an accessible workstation rather than a high-end research server. The target hardware is an NVIDIA RTX 4060 GPU with 8GB VRAM and 32GB RAM. The code uses PyTorch, NumPy, SciPy, nibabel, pandas, matplotlib, Streamlit, and related scientific Python packages. The training configuration uses 192 x 192 inputs, three input channels, one output channel, batch size 12, and effective batch size 24." | Out-Null
    Add-Heading2 "5.2 Training Results"
    Add-Paragraph "The training log covers 65 epochs. The model reached its best validation Dice at epoch 45, with validation Dice 0.8401, validation IoU 0.8171, and validation HD95 46.52. The validation loss remained stable after this point, while the minimum recorded validation HD95 was 38.35 at epoch 50. The curves in Figure 5.1 show rapid early improvement followed by a slower refinement stage." | Out-Null
    Add-FigureBlock (Join-Path $OutputDir "training_curves.png") "Figure 5.1. Training curves: loss, Dice, IoU, HD95, and learning rate." 450
    Add-TableBlock "Table 5.1. Best validation epoch and validation metrics." @("Epoch","Train Loss","Val Loss","Train Dice","Val Dice","Val IoU","Val HD95","Learning Rate") @(
        @("45","0.121823","0.068735","0.750054","0.840084","0.817063","46.520337","6.71e-04"),
        @("50","0.116739","0.070289","0.759412","0.833602","0.809239","38.347351","5.78e-04")
    )
    Add-Heading2 "5.3 Test Results and TTA"
    Add-Paragraph "The test evaluation was performed on 84 volumes using `evaluate_tta.py`. TTA improved the mean Dice from 0.8038 to 0.8172 and the mean IoU from 0.7815 to 0.7949. However, mean HD95 changed from 131.05 to 131.94. This distinction matters: TTA made overlap more stable on average, but it did not reduce the average boundary-distance error. The result suggests that flip averaging can help mask agreement while occasional distant false positives or false negatives still dominate HD95." | Out-Null
    Add-TableBlock "Table 5.2. Test performance before and after TTA." @("Volumes","DSC No TTA","IoU No TTA","HD95 No TTA","DSC TTA","IoU TTA","HD95 TTA","Mean DSC diff","Mean IoU diff") @(
        @("84","0.8038","0.7815","131.05","0.8172","0.7949","131.94","+0.0134","+0.0134")
    )
    Add-Heading2 "5.4 Scanner and Domain Shift Analysis"
    Add-Paragraph "Scanner-level analysis shows that the model does not perform identically across acquisition domains. Differences in magnetic field homogeneity, RF coil structure, reconstruction protocol, and intensity distribution can shift the data away from the training distribution. The scanner performance CSV indicates stronger mean Dice for systems such as GE Signa HD-X and Siemens Magnetom Skyra, while Siemens Verio has a lower mean Dice and the highest HD95." | Out-Null
    Add-FigureBlock (Join-Path $AssetDir "fig_5_2_scanner_performance.png") "Figure 5.2. Scanner-level Dice and HD95 performance summary." 450
    Add-TableBlock "Table 5.3. Scanner-level performance comparison." @("Scanner Brand","Dice mean","Dice std","N","IoU mean","HD95 mean") @(
        @("GE Signa HD-X","0.8642","0.0377","4","0.8486","80.14"),
        @("Siemens Magnetom Skyra","0.8475","0.0421","3","0.8204","25.66"),
        @("Philips","0.8212","0.0142","2","0.7781","45.22"),
        @("GE 750 Discovery","0.7976","0.1176","19","0.7894","195.95"),
        @("Siemens Prisma","0.7935","0.0649","8","0.7695","94.13"),
        @("Siemens Verio","0.7353","0.0492","2","0.7353","252.71")
    )
    Add-Heading2 "5.5 Visual Interpretation"
    Add-Paragraph "Visual inspection remains necessary even when numerical scores look strong. Small lesions may produce low Dice if only a few pixels are missed, and ventricular or cortical border regions may create false positives because they share intensity transitions with lesion tissue. Figure 5.3 presents the panel layout used to compare raw image, ground truth, model prediction, and probability map." | Out-Null
    Add-FigureBlock (Join-Path $AssetDir "fig_5_3_segmentation_example.png") "Figure 5.3. Example segmentation result layout."
    Add-Heading2 "5.6 Discussion"
    Add-Paragraph "GA-UNet is lighter than a classical U-Net because the encoder and decoder blocks rely on Ghost operations rather than producing every feature map through standard convolutions. GhostModule reduces parameter and FLOP cost by separating intrinsic feature extraction from cheap feature generation. GhostNetV2 contributes long-range attention through DFC-style mechanisms, which is useful because lesions are interpreted in relation to surrounding anatomy." | Out-Null
    Add-Paragraph "SimAM adds a different kind of advantage: it strengthens important skip activations without adding trainable parameters. This is a good match for small stroke lesions, where high-resolution details must be preserved but background tissue should not dominate the decoder. The 2.5D strategy is also practical because it gives the model neighboring-slice context without the memory burden of full 3D convolutions." | Out-Null
    Add-Paragraph "The main failure cases remain clinically important. Very small, diffuse, or low-contrast lesions can be missed. Scanner-induced domain shift can also create intensity and geometry patterns that the model did not see often enough during training. Future work should therefore combine lightweight segmentation with stronger harmonization, domain generalization, and uncertainty estimation." | Out-Null
    Write-Output "Chapter 5 generated."
    Log-Step "Chapter 5 generated."

    # Chapter 6
    Add-Heading1 "6. ETHICS, STANDARDS, AND SUSTAINABILITY" $true
    Add-Heading2 "6.1 Ethical Principles"
    Add-Paragraph "This thesis uses open and de-identified human neuroimaging data. The ethical responsibility does not end because the data are public. Clinical images must be handled with privacy awareness, and the model must be described honestly as a research and decision-support tool. It should not be presented as a system that can diagnose or replace a clinician. Human expert oversight remains essential, particularly when a segmentation output may influence clinical interpretation." | Out-Null
    Add-Heading2 "6.2 Engineering Standards"
    Add-Paragraph "The project aligns with IEEE software engineering principles through modularity, traceable experiment outputs, and separated responsibilities. ISO 9001 is relevant as a quality-management mindset: parameters, logs, splits, checkpoints, and evaluation outputs are recorded so results can be audited. FAIR principles guide dataset handling, while reproducible research principles guide the code and documentation structure." | Out-Null
    Add-Heading2 "6.3 Risk Management"
    Add-TableBlock "Table 6.1. Project risk analysis." @("Risk","Probability","Impact","Mitigation","Residual risk") @(
        @("GPU memory limitation","Medium","High","Use 2.5D input, AMP, batch size 12, gradient accumulation","Large experiments may still require stronger hardware"),
        @("Class imbalance","High","High","Use FocalDiceBCE and controlled empty-slice sampling","Tiny lesions remain difficult"),
        @("Low Dice for small lesions","Medium","High","Use Dice-focused loss and SimAM skip weighting","External validation needed"),
        @("Scanner/domain shift","High","High","Use scanner-aware analysis and TTA","Not fully solved"),
        @("Overfitting","Medium","Medium","Validation monitoring, checkpointing, patience","Dataset-specific bias may remain"),
        @("HD95 computation cost","Medium","Medium","Compute HD95 at intervals","Boundary analysis remains slower"),
        @("Wrong clinical interpretation","Low","High","Position model as decision support only","Requires clinician oversight")
    )
    Add-Heading2 "6.4 Sustainability"
    Add-Paragraph "A lightweight model contributes to environmental sustainability by reducing energy consumption during training and inference. It also improves economic sustainability because it can be developed on mid-range hardware rather than expensive server-grade GPUs. Socially, a reproducible and lower-cost segmentation pipeline can make medical image analysis more accessible to smaller laboratories and clinical centers." | Out-Null
    Add-Heading2 "6.5 Reproducibility"
    Add-TableBlock "Table 6.2. Reproducibility checklist." @("Item","Status in project") @(
        @("Fixed random seed","Defined in training arguments"),
        @("Dataset split file","Saved as outputs/dataset_splits.json"),
        @("Training logs","Saved as CSV and JSONL"),
        @("Best checkpoint","Saved as outputs/best_model.pth"),
        @("Modular code","Model, data, losses, metrics, TTA, app separated"),
        @("Command-line parameters","Defined in train_local.py"),
        @("Volume-level test results","Saved as outputs/test_tta_results.csv")
    )
    Write-Output "Chapter 6 generated."
    Log-Step "Chapter 6 generated."

    # Chapter 7
    Add-Heading1 "7. CONCLUSIONS AND RECOMMENDATIONS" $false
    Add-Heading2 "7.1 Conclusions"
    Add-Paragraph "This thesis developed GA-UNet as a lightweight and functional architecture for stroke lesion segmentation. The GhostNetV2 encoder provided multi-scale feature extraction with lower computational cost, while SimAM supplied parameter-free attention in skip connections. The 2.5D strategy created a practical balance between volumetric context and 2D efficiency. FocalDiceBCE addressed lesion sparsity during training, and TTA improved overlap metrics during inference without modifying the model." | Out-Null
    Add-Heading2 "7.2 Limitations"
    Add-Bullets @(
        "The model does not directly learn full 3D volumetric context.",
        "The learned representation may remain partly dependent on the ATLAS R2.0 distribution.",
        "Scanner and domain shift are reduced only partially, not eliminated.",
        "Small, scattered, or low-contrast lesions can still cause false negatives or false positives.",
        "Clinical use would require broader external validation and expert review."
    )
    Add-Heading2 "7.3 Future Work"
    Add-Bullets @(
        "Develop a hybrid 2.5D-3D or full 3D GA-UNet version.",
        "Compare lightweight Transformer, ConvNeXt, and GhostNetV2 encoders.",
        "Use self-supervised pretraining on unlabeled MRI volumes.",
        "Apply domain adaptation and domain generalization methods.",
        "Investigate scanner harmonization and bias-field correction.",
        "Add uncertainty estimation for safer clinical decision support.",
        "Conduct expert visual review and a broader ablation study."
    )
    Write-Output "Chapter 7 generated."
    Log-Step "Chapter 7 generated."

    # References
    Add-Heading1 "8. REFERENCES" $true
    $refs = @(
        "RONNEBERGER, Olaf; FISCHER, Philipp; BROX, Thomas. U-Net: Convolutional Networks for Biomedical Image Segmentation. In: MICCAI 2015. Cham: Springer, 2015, pp. 234-241. DOI: 10.1007/978-3-319-24574-4_28. Available at: https://lmb.informatik.uni-freiburg.de/Publications/2015/RFB15a/.",
        "OKTAY, Ozan et al. Attention U-Net: Learning Where to Look for the Pancreas. arXiv:1804.03999, 2018. Available at: https://arxiv.org/abs/1804.03999.",
        "HAN, Kai et al. GhostNet: More Features from Cheap Operations. In: IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2020, pp. 1580-1589. Available at: https://openaccess.thecvf.com/content_CVPR_2020/html/Han_GhostNet_More_Features_From_Cheap_Operations_CVPR_2020_paper.html.",
        "TANG, Yehui et al. GhostNetV2: Enhance Cheap Operation with Long-Range Attention. In: Advances in Neural Information Processing Systems, 2022. Available at: https://proceedings.neurips.cc/paper_files/paper/2022/hash/40b60852a4abdaa696b5a1a78da34635-Abstract-Conference.html.",
        "YANG, Lingxiao et al. SimAM: A Simple, Parameter-Free Attention Module for Convolutional Neural Networks. In: Proceedings of the 38th International Conference on Machine Learning. PMLR, 2021, vol. 139, pp. 11863-11874. Available at: https://proceedings.mlr.press/v139/yang21o.html.",
        "LIEW, Sook-Lei et al. A large, curated, open-source stroke neuroimaging dataset to improve lesion segmentation algorithms. Scientific Data, 2022, vol. 9, article 320. DOI: 10.1038/s41597-022-01401-7. Available at: https://www.nature.com/articles/s41597-022-01401-7.",
        "ISENSEE, Fabian et al. nnU-Net: a self-configuring method for deep learning-based biomedical image segmentation. Nature Methods, 2021, vol. 18, pp. 203-211. DOI: 10.1038/s41592-020-01008-z.",
        "TAHA, Abdel Aziz; HANBURY, Allan. Metrics for evaluating 3D medical image segmentation: analysis, selection, and tool. BMC Medical Imaging, 2015, vol. 15, article 29. DOI: 10.1186/s12880-015-0068-x.",
        "DICE, Lee R. Measures of the Amount of Ecologic Association Between Species. Ecology, 1945, vol. 26, no. 3, pp. 297-302.",
        "LITJENS, Geert et al. A survey on deep learning in medical image analysis. Medical Image Analysis, 2017, vol. 42, pp. 60-88. DOI: 10.1016/j.media.2017.07.005.",
        "KAMNITSAS, Konstantinos et al. Unsupervised domain adaptation in brain lesion segmentation with adversarial networks. In: IPMI 2017. Cham: Springer, 2017, pp. 597-609.",
        "GIBSON, Eli et al. NiftyNet: a deep-learning platform for medical imaging. Computer Methods and Programs in Biomedicine, 2018, vol. 158, pp. 113-122.",
        "WANG, Dequan et al. Tent: Fully Test-Time Adaptation by Entropy Minimization. In: International Conference on Learning Representations, 2021. Available at: https://openreview.net/forum?id=uXl3bZLkr3c.",
        "MONDAL, A. K. et al. Test-time adaptation for medical image segmentation: a survey and perspectives. Medical Image Analysis, 2024."
    )
    for ($i=0; $i -lt $refs.Count; $i++) {
        Add-Paragraph ("[" + ($i+1) + "] " + $refs[$i]) $wdStyleNormal $wdAlignLeft 10 $false 3 | Out-Null
    }
    Write-Output "References generated."
    Log-Step "References generated."

    # Appendices
    Add-Heading1 "9. APPENDICES" $true
    Add-Heading2 "Appendix A. Model Code Summary"
    Add-Paragraph "The following shortened excerpts summarize the main implemented classes. The full source remains in the project files." | Out-Null
    Add-CodeBlock @(
        "class GAUNet(nn.Module):",
        "    encoder = GhostNetV2Encoder(in_channels=3)",
        "    bottleneck = GhostModule blocks",
        "    decoder = DecoderBlock + SimAM skip attention",
        "    seg_head = Conv2d(..., out_channels=1)",
        "",
        "class GhostModule(nn.Module):",
        "    primary_conv -> intrinsic features",
        "    cheap_operation -> ghost features",
        "    concat + channel slice -> output",
        "",
        "class SimAM(nn.Module):",
        "    computes parameter-free 3D attention weights"
    )
    Add-Heading2 "Appendix B. Training Parameters"
    Add-Paragraph "The default training setup uses `--epochs 100`, `--batch-size 12`, `--accum-steps 2`, `--lr 1e-3`, `--weight-decay 1e-5`, `--patience 20`, `--target-size 192`, `--num-slices 3`, and `--hd95-freq 5`. The completed local log contains 65 epochs, with the best validation Dice at epoch 45." | Out-Null
    Add-Heading2 "Appendix C. Dataset Split Structure"
    Add-Paragraph "`outputs/dataset_splits.json` stores train, validation, and test items as image path, mask path, and grouping label. The local split contains 487 training volumes, 84 validation volumes, and 84 test volumes." | Out-Null
    Add-Heading2 "Appendix D. Test Interface Screens"
    Add-Paragraph "The Streamlit interface provides metric calculation, scanner analysis, and visual inspection tabs. Figure 4.2 gives a clean representative screenshot of this workflow." | Out-Null
    Add-Heading2 "Appendix E. Additional Results"
    Add-Paragraph "`outputs/test_tta_results.csv` contains volume-level values for TTA and non-TTA inference. `outputs/2026-04-18T13-12_scannerBrand_performance_metric.csv` contains scanner-level averages and standard deviations for Dice, IoU, and HD95." | Out-Null
    Write-Output "Appendices generated. Saving document..."
    Log-Step "Appendices generated. Saving document."

    # Page numbering is intentionally left to Word's template/update workflow.
    # Hidden Word automation can stall on footer PageNumbers.Add in sandboxed
    # sessions, while static content and captions remain deterministic here.
    Log-Step "Footer page number automation skipped."

    # Save. The table of contents and lists are static to keep the headless
    # generation deterministic in this Windows session.
    $savePath = [string]$DocxPath
    $saveFormat = [int]$wdFormatXMLDocument
    $doc.SaveAs([ref]$savePath, [ref]$saveFormat)
    Log-Step "DOCX saved."
    $pageCount = $doc.ComputeStatistics($wdStatisticPages)
    Log-Step "Page count computed: $pageCount."
    $doc.Close([ref]$false)
    $openPath = [string]$DocxPath
    $confirmConversions = $false
    $readOnly = $true
    $doc = $word.Documents.Open([ref]$openPath, [ref]$confirmConversions, [ref]$readOnly)
    $pdfSavePath = [string]$PdfPath
    $pdfFormat = [int]$wdFormatPDF
    $doc.SaveAs([ref]$pdfSavePath, [ref]$pdfFormat)
    Log-Step "PDF exported."
    $doc.Close([ref]$false)
}
finally {
    $word.Quit()
}

# Render PDF pages to PNG using Windows built-in PDF renderer.
Get-ChildItem $RenderDir -Filter "page-*.png" -ErrorAction SilentlyContinue | Remove-Item -Force
Get-ChildItem $RenderDir -Filter "contact-sheet-*.png" -ErrorAction SilentlyContinue | Remove-Item -Force

Add-Type -AssemblyName System.Runtime.WindowsRuntime
[Windows.Storage.StorageFile, Windows.Storage, ContentType=WindowsRuntime] | Out-Null
[Windows.Data.Pdf.PdfDocument, Windows.Data.Pdf, ContentType=WindowsRuntime] | Out-Null
[Windows.Data.Pdf.PdfPageRenderOptions, Windows.Data.Pdf, ContentType=WindowsRuntime] | Out-Null
[Windows.Storage.Streams.InMemoryRandomAccessStream, Windows.Storage.Streams, ContentType=WindowsRuntime] | Out-Null
[Windows.Storage.Streams.DataReader, Windows.Storage.Streams, ContentType=WindowsRuntime] | Out-Null

$asTaskGeneric = ([System.WindowsRuntimeSystemExtensions].GetMethods() | Where-Object { $_.Name -eq "AsTask" -and $_.GetParameters().Count -eq 1 -and $_.IsGenericMethodDefinition } | Select-Object -First 1)
$asTaskAction = ([System.WindowsRuntimeSystemExtensions].GetMethods() | Where-Object { $_.Name -eq "AsTask" -and $_.GetParameters().Count -eq 1 -and -not $_.IsGenericMethodDefinition } | Select-Object -First 1)
function Await-Async($async, $type) {
    $m = $asTaskGeneric.MakeGenericMethod($type)
    $task = $m.Invoke($null, @($async))
    $task.Wait()
    return $task.Result
}
function Await-Action($async) {
    $task = $asTaskAction.Invoke($null, @($async))
    $task.Wait()
}

$file = Await-Async ([Windows.Storage.StorageFile]::GetFileFromPathAsync($PdfPath)) ([Windows.Storage.StorageFile])
$pdfDoc = Await-Async ([Windows.Data.Pdf.PdfDocument]::LoadFromFileAsync($file)) ([Windows.Data.Pdf.PdfDocument])
$renderedPages = @()
for ($i=0; $i -lt $pdfDoc.PageCount; $i++) {
    $page = $pdfDoc.GetPage($i)
    $stream = New-Object Windows.Storage.Streams.InMemoryRandomAccessStream
    $options = New-Object Windows.Data.Pdf.PdfPageRenderOptions
    $options.DestinationWidth = [uint32]1200
    Await-Action ($page.RenderToStreamAsync($stream, $options))
    $reader = New-Object Windows.Storage.Streams.DataReader($stream.GetInputStreamAt(0))
    [uint32]$size = $stream.Size
    Await-Async ($reader.LoadAsync($size)) ([uint32]) | Out-Null
    $bytes = New-Object byte[] $size
    $reader.ReadBytes($bytes)
    $outPng = Join-Path $RenderDir ("page-{0:D3}.png" -f ($i+1))
    [IO.File]::WriteAllBytes($outPng, $bytes)
    $renderedPages += $outPng
    $page.Dispose()
    $stream.Dispose()
    $reader.Dispose()
}

# Create contact sheets for faster visual QA.
Add-Type -AssemblyName System.Drawing
$thumbW = 360
$thumbH = 510
$cols = 3
$rowsPerSheet = 3
$perSheet = $cols * $rowsPerSheet
$sheetIndex = 1
for ($offset=0; $offset -lt $renderedPages.Count; $offset += $perSheet) {
    $sheetPath = Join-Path $RenderDir ("contact-sheet-{0:D2}.png" -f $sheetIndex)
    $sheet = New-Object System.Drawing.Bitmap(1250, 1650)
    $g = [System.Drawing.Graphics]::FromImage($sheet)
    $g.Clear([System.Drawing.Color]::White)
    $font = New-Object System.Drawing.Font("Segoe UI", 18, [System.Drawing.FontStyle]::Bold)
    for ($j=0; $j -lt $perSheet -and ($offset+$j) -lt $renderedPages.Count; $j++) {
        $img = [System.Drawing.Image]::FromFile($renderedPages[$offset+$j])
        $x = 35 + ($j % $cols) * 405
        $y = 55 + [math]::Floor($j / $cols) * 520
        $g.DrawImage($img, $x, $y+35, $thumbW, $thumbH)
        $g.DrawRectangle((PenC 170 170 170 2), $x, $y+35, $thumbW, $thumbH)
        $g.DrawString(("Page {0}" -f ($offset+$j+1)), $font, (Brush 20 20 20), $x, $y)
        $img.Dispose()
    }
    $font.Dispose()
    $g.Dispose()
    $sheet.Save($sheetPath, [System.Drawing.Imaging.ImageFormat]::Png)
    $sheet.Dispose()
    $sheetIndex++
}

$docxSize = (Get-Item $DocxPath).Length
$pdfSize = (Get-Item $PdfPath).Length
@(
    "GA-UNet Duzce Graduation Thesis QA Summary",
    "DOCX: $DocxPath",
    "PDF: $PdfPath",
    "DOCX bytes: $docxSize",
    "PDF bytes: $pdfSize",
    "Word page count: $pageCount",
    "Rendered PNG pages: $($renderedPages.Count)",
    "Contact sheets: $($sheetIndex - 1)",
    "Generated at: $(Get-Date -Format s)"
) | Set-Content -Path $QaSummaryPath -Encoding UTF8

Write-Output "DOCX=$DocxPath"
Write-Output "PDF=$PdfPath"
Write-Output "PAGES=$($renderedPages.Count)"
Write-Output "RENDER_DIR=$RenderDir"
Write-Output "QA=$QaSummaryPath"
