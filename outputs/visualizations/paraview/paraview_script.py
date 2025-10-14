from paraview.simple import *
import os

paraview.simple._DisableFirstRenderCameraReset()

print("Loading gravitational field data...")

data_vtx_dir = 'outputs/data/vtx'
data_xdmf_dir = 'outputs/data/xdmf'

potential_bp = os.path.join(data_vtx_dir, 'gravitational_field_potential.bp')
scalar_bp = os.path.join(data_vtx_dir, 'gravitational_field_scalar_fields.bp')
vector_bp = os.path.join(data_vtx_dir, 'gravitational_field_vector_field.bp')
xdmf_path = os.path.join(data_xdmf_dir, 'gravitational_field.xdmf')

reader = None

if os.path.exists(scalar_bp):
    print(f"Loading scalar fields from {scalar_bp}")
    reader = VTXReader(FileName=[scalar_bp])
elif os.path.exists(potential_bp):
    print(f"Loading potential field from {potential_bp}")
    reader = VTXReader(FileName=[potential_bp])
elif os.path.exists(vector_bp):
    print(f"Loading vector field from {vector_bp}")
    reader = VTXReader(FileName=[vector_bp])
elif os.path.exists(xdmf_path):
    print(f"Loading XDMF data from {xdmf_path}")
    reader = XDMFReader(FileNames=[xdmf_path])
else:
    print("ERROR: No data files found. Run the simulation first.")
    exit(1)

renderView = GetActiveViewOrCreate('RenderView')

display = Show(reader, renderView)
display.Representation = 'Surface'

try:
    display.ColorArrayName = ['CELLS', 'field_magnitude']
    ColorBy(display, ('CELLS', 'field_magnitude'))
except:
    try:
        display.ColorArrayName = ['POINTS', 'field_magnitude']
        ColorBy(display, ('POINTS', 'field_magnitude'))
    except:
        print("Note: Could not find field_magnitude, using default coloring")

try:
    magnitudeLUT = GetColorTransferFunction('field_magnitude')
    magnitudeLUT.ApplyPreset('Plasma (matplotlib)', True)
    magnitudePWF = GetOpacityTransferFunction('field_magnitude')
    display.SetScalarBarVisibility(renderView, True)
except:
    print("Note: Could not set color map")

renderView.ResetCamera()
renderView.OrientationAxesVisibility = 1

print("Creating slice filter...")
slice1 = Slice(Input=reader)
slice1.SliceType = 'Plane'
slice1.SliceType.Origin = [0.0, 0.0, 0.0]
slice1.SliceType.Normal = [0.0, 0.0, 1.0]

sliceDisplay = Show(slice1, renderView)
sliceDisplay.Representation = 'Surface'

try:
    ColorBy(sliceDisplay, ('POINTS', 'energy_density'))
    Hide(reader, renderView)
    energyLUT = GetColorTransferFunction('energy_density')
    energyLUT.ApplyPreset('Inferno (matplotlib)', True)
    energyLUT.RescaleTransferFunction(0.0, 1.0)
    sliceDisplay.SetScalarBarVisibility(renderView, True)
except:
    try:
        ColorBy(sliceDisplay, ('POINTS', 'field_magnitude'))
        Hide(reader, renderView)
        magnitudeLUT = GetColorTransferFunction('field_magnitude')
        magnitudeLUT.ApplyPreset('Inferno (matplotlib)', True)
        magnitudeLUT.RescaleTransferFunction(0.0, 1.0)
        sliceDisplay.SetScalarBarVisibility(renderView, True)
    except:
        print("Note: Could not set slice coloring")

print("Setup complete!")
print("Available fields depend on which file you opened:")
print("  - gravitational_field_potential.bp:    gravitational_potential")
print("  - gravitational_field_scalar_fields.bp: field_magnitude, energy_density, curvature_scalar")
print("  - gravitational_field_vector_field.bp:  field_strength")
print("  - gravitational_field.xdmf:            all fields")

renderView.ResetCamera()
Render()
