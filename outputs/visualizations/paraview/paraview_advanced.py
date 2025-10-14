from paraview.simple import *
import os

paraview.simple._DisableFirstRenderCameraReset()

print("=" * 70)
print("ADVANCED PARAVIEW VISUALIZATION SETUP")
print("=" * 70)

print("\nLoading gravitational field data...")

# Define directory structure
DATA_VTX_DIR = "outputs/data/vtx"
DATA_XDMF_DIR = "outputs/data/xdmf"

# Define file paths
potential_bp = os.path.join(DATA_VTX_DIR, 'gravitational_field_potential.bp')
scalar_bp = os.path.join(DATA_VTX_DIR, 'gravitational_field_scalar_fields.bp')
vector_bp = os.path.join(DATA_VTX_DIR, 'gravitational_field_vector_field.bp')
xdmf_path = os.path.join(DATA_XDMF_DIR, 'gravitational_field.xdmf')

readers = []

if os.path.exists(potential_bp):
    potential_reader = VTXReader(FileName=[potential_bp])
    potential_reader.UpdatePipeline()
    readers.append(potential_reader)
    print(f"Loaded potential field from {potential_bp}")

if os.path.exists(scalar_bp):
    scalar_reader = VTXReader(FileName=[scalar_bp])
    scalar_reader.UpdatePipeline()
    readers.append(scalar_reader)
    print(f"Loaded scalar fields from {scalar_bp}")

if os.path.exists(vector_bp):
    vector_reader = VTXReader(FileName=[vector_bp])
    vector_reader.UpdatePipeline()
    readers.append(vector_reader)
    print(f"Loaded vector field from {vector_bp}")

if not readers and os.path.exists(xdmf_path):
    xdmf_reader = XDMFReader(FileNames=[xdmf_path])
    xdmf_reader.UpdatePipeline()
    readers.append(xdmf_reader)
    print(f"Loaded fields from {xdmf_path}")

if not readers:
    raise RuntimeError('No field data files found. Run the simulation first.')

reader = readers[0]

renderView = CreateView('RenderView')
renderView.ViewSize = [1920, 1080]
renderView.Background = [0.1, 0.1, 0.15]
renderView.OrientationAxesVisibility = 1

print("Creating multi-panel visualization...")

slice_z = Slice(Input=reader)
slice_z.SliceType = 'Plane'
slice_z.SliceType.Origin = [0.0, 0.0, 0.0]
slice_z.SliceType.Normal = [0.0, 0.0, 1.0]

sliceDisplay = Show(slice_z, renderView)
sliceDisplay.Representation = 'Surface'
ColorBy(sliceDisplay, ('POINTS', 'field_magnitude'))

magnitudeLUT = GetColorTransferFunction('field_magnitude')
magnitudeLUT.ApplyPreset('Plasma (matplotlib)', True)
magnitudeLUT.RescaleTransferFunction(0.0, 0.5)

magnitudePWF = GetOpacityTransferFunction('field_magnitude')

sliceDisplay.SetScalarBarVisibility(renderView, True)

print("Adding contour visualization...")
contour = Contour(Input=reader)
contour.ContourBy = ['POINTS', 'gravitational_potential']
contour.Isosurfaces = [-1.0, -0.8, -0.6, -0.4, -0.2, -0.1]
contour.PointMergeMethod = 'Uniform Binning'

contourDisplay = Show(contour, renderView)
contourDisplay.Representation = 'Surface'
contourDisplay.Opacity = 0.3
ColorBy(contourDisplay, ('POINTS', 'energy_density'))

energyLUT = GetColorTransferFunction('energy_density')
energyLUT.ApplyPreset('Viridis (matplotlib)', True)

print("Adding vector field visualization...")
calculator = Calculator(Input=reader)
calculator.ResultArrayName = 'normalized_field'
calculator.Function = 'field_strength / (mag(field_strength) + 0.001)'

threshold = Threshold(Input=calculator)
threshold.Scalars = ['CELLS', 'field_magnitude']
threshold.ThresholdRange = [0.01, 1.0]

resampleToImage = ResampleToImage(Input=threshold)
resampleToImage.SamplingDimensions = [20, 20, 20]

glyph = Glyph(Input=resampleToImage, GlyphType='Arrow')
glyph.OrientationArray = ['POINTS', 'normalized_field']
glyph.ScaleArray = ['POINTS', 'field_magnitude']
glyph.ScaleFactor = 2.0
glyph.GlyphMode = 'All Points'

glyphDisplay = Show(glyph, renderView)
glyphDisplay.Representation = 'Surface'
ColorBy(glyphDisplay, ('POINTS', 'field_magnitude'))
glyphDisplay.SetScalarBarVisibility(renderView, False)

print("Creating streamlines...")
streamTracer = StreamTracer(Input=calculator, SeedType='Point Cloud')
streamTracer.Vectors = ['POINTS', 'normalized_field']
streamTracer.MaximumStreamlineLength = 20.0

streamTracer.SeedType.Center = [0.0, 0.0, 0.0]
streamTracer.SeedType.Radius = 10.0
streamTracer.SeedType.NumberOfPoints = 100

streamDisplay = Show(streamTracer, renderView)
streamDisplay.Representation = 'Surface'
ColorBy(streamDisplay, ('POINTS', 'field_magnitude'))

streamLUT = GetColorTransferFunction('field_magnitude')
streamLUT.ApplyPreset('Cool to Warm', True)

streamDisplay.SetScalarBarVisibility(renderView, True)

print("Adjusting camera...")
renderView.ResetCamera()
camera = GetActiveCamera()
camera.Elevation(30)
camera.Azimuth(45)

print("\n" + "=" * 70)
print("VISUALIZATION COMPLETE")
print("=" * 70)

print("\nCreated visualizations:")
print("  1. Slice (Z=0) - Shows field magnitude in XY plane")
print("  2. Contours - Equipotential surfaces")
print("  3. Glyphs - Vector field arrows")
print("  4. Streamlines - Field lines from central region")

print("\nInteractive controls:")
print("  - Toggle visibility: Click eye icon in Pipeline Browser")
print("  - Adjust time: Use time slider at top")
print("  - Change colors: Edit color map in Color Map Editor")
print("  - Modify filters: Select in Pipeline Browser and adjust Properties")

print("\nRecommended next steps:")
print("  1. Hide/show different visualizations to explore")
print("  2. Adjust contour isovalues")
print("  3. Change glyph scale factor")
print("  4. Modify streamline seed positions")
print("  5. Try different color maps")

Render()

print("\n" + "=" * 70)

