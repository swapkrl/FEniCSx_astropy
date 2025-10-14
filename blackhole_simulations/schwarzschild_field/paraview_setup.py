import os

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "outputs", "visualizations", "paraview")
DATA_VTX_DIR = os.path.join(SCRIPT_DIR, "outputs", "data", "vtx")
DATA_XDMF_DIR = os.path.join(SCRIPT_DIR, "outputs", "data", "xdmf")

def ensure_dir(path):
    os.makedirs(path, exist_ok=True)
    return path

def create_paraview_state():
    ensure_dir(OUTPUT_DIR)
    state_path = os.path.join(OUTPUT_DIR, "paraview_state.pvsm")
    
    state_content = f"""<?xml version="1.0"?>
<ParaViewState version="5.11.0">
  <Proxy group="sources" type="VTXReader" id="250" servers="1">
    <Property name="FileName" id="250.FileName" number_of_elements="1">
      <Element index="0" value="{os.path.join(DATA_VTX_DIR, 'gravitational_field_scalar_fields.bp')}"/>
    </Property>
  </Proxy>
</ParaViewState>"""
    
    with open(state_path, "w") as f:
        f.write(state_content)
    
    print(f"ParaView state file created: {state_path}")
    print("To use: Open ParaView and load this state file via File → Load State")

def create_paraview_python_script():
    ensure_dir(OUTPUT_DIR)
    script_path = os.path.join(OUTPUT_DIR, "paraview_script.py")
    
    script_content = f"""from paraview.simple import *
import os

paraview.simple._DisableFirstRenderCameraReset()

print("Loading gravitational field data...")

data_vtx_dir = '{DATA_VTX_DIR}'
data_xdmf_dir = '{DATA_XDMF_DIR}'

potential_bp = os.path.join(data_vtx_dir, 'gravitational_field_potential.bp')
scalar_bp = os.path.join(data_vtx_dir, 'gravitational_field_scalar_fields.bp')
vector_bp = os.path.join(data_vtx_dir, 'gravitational_field_vector_field.bp')
xdmf_path = os.path.join(data_xdmf_dir, 'gravitational_field.xdmf')

reader = None

if os.path.exists(scalar_bp):
    print(f"Loading scalar fields from {{scalar_bp}}")
    reader = VTXReader(FileName=[scalar_bp])
elif os.path.exists(potential_bp):
    print(f"Loading potential field from {{potential_bp}}")
    reader = VTXReader(FileName=[potential_bp])
elif os.path.exists(vector_bp):
    print(f"Loading vector field from {{vector_bp}}")
    reader = VTXReader(FileName=[vector_bp])
elif os.path.exists(xdmf_path):
    print(f"Loading XDMF data from {{xdmf_path}}")
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
"""
    
    with open(script_path, "w") as f:
        f.write(script_content)
    
    print(f"ParaView Python script created: {script_path}")
    print(f"To use: In ParaView, go to Tools → Python Shell → Run Script → Select {script_path}")

def create_quick_start_guide():
    ensure_dir(OUTPUT_DIR)
    guide_path = os.path.join(OUTPUT_DIR, "PARAVIEW_GUIDE.md")
    
    guide_content = """# ParaView Quick Start Guide

## Method 1: Manual Loading (Recommended for Beginners)

1. Open ParaView
2. File → Open → Select one of the following:
   - `outputs/data/vtx/gravitational_field_potential.bp` (for potential field)
   - `outputs/data/vtx/gravitational_field_scalar_fields.bp` (for magnitude, energy density, curvature)
   - `outputs/data/vtx/gravitational_field_vector_field.bp` (for vector field)
   - `outputs/data/xdmf/gravitational_field.xdmf` (for all fields in one file)
3. Click 'Apply' in the Properties panel
4. In the toolbar, change 'Solid Color' dropdown to a field name
5. Adjust color map using the Color Map Editor

## Method 2: Using Python Script (Automated Setup)

1. Open ParaView
2. Tools → Python Shell
3. Click 'Run Script' button
4. Select 'outputs/visualizations/paraview/paraview_script.py' (basic) or 'outputs/visualizations/paraview/paraview_advanced.py' (multi-view)
5. The visualization will be automatically configured

## Available Visualization Fields

Each field is stored in a separate file for better compatibility:

- **outputs/data/vtx/gravitational_field_potential.bp**:
  - `gravitational_potential`: The gravitational potential (Φ)

- **outputs/data/vtx/gravitational_field_scalar_fields.bp**:
  - `field_magnitude`: |∇Φ| - Magnitude of gravitational force
  - `energy_density`: ½|∇Φ|² - Energy density distribution  
  - `curvature_scalar`: Curvature-like quantity (√(∇²Φ)²)

- **outputs/data/vtx/gravitational_field_vector_field.bp**:
  - `field_strength`: 3D vector field -∇Φ (gravitational force direction)

- **outputs/data/xdmf/gravitational_field.xdmf**:
  - All fields in one file (may be slower but more convenient)

## Recommended Visualization Techniques

### Volume Rendering
- Open `outputs/data/vtx/gravitational_field_scalar_fields.bp`
- Select field: 'field_magnitude' or 'energy_density'
- Representation: Volume
- Adjust opacity in Color Map Editor for transparency
- Best for: Overall structure and density distribution

### Slice View
- Filter → Slice
- Origin: [0, 0, 0]
- Normal: [0, 0, 1] for XY plane
- Color by: 'energy_density' or 'field_magnitude'
- Best for: Cross-sectional analysis

### Contour/Isosurface
- Open `outputs/data/vtx/gravitational_field_potential.bp`
- Filter → Contour
- Contour By: 'gravitational_potential'
- Add 5-10 isovalues
- Best for: Equipotential surfaces

### Vector Field Arrows
- Open `outputs/data/vtx/gravitational_field_vector_field.bp`
- Filter → Glyph
- Glyph Type: Arrow
- Vectors: 'field_strength'
- Scale Mode: 'vector'
- Best for: Field direction visualization

### Streamlines
- Open `outputs/data/vtx/gravitational_field_vector_field.bp`
- Filter → Stream Tracer
- Vectors: 'field_strength'
- Seed Type: Point Cloud or Line
- Best for: Field line visualization

## Tips

1. **Color Maps**: Use 'Plasma', 'Viridis', or 'Inferno' for scientific data
2. **Rescale**: Right-click color bar → 'Rescale to Data Range'
3. **Logarithmic Scale**: Edit color map → Use log scale for high dynamic range
4. **Camera**: Use mouse to rotate (left), pan (middle), zoom (right/scroll)
5. **Animation**: Use time controls at top to animate through timesteps
6. **Save**: File → Save Screenshot or Save Animation
7. **Multiple Files**: You can open multiple BP files at once to visualize different fields

## Troubleshooting

**Problem: Nothing visible**
- Solution: Change color map field, rescale to data range, adjust opacity

**Problem: Data looks flat**
- Solution: Use volume rendering or add multiple slices at different positions

**Problem: Colors washed out**
- Solution: Use logarithmic color scale or rescale to custom range

**Problem: BP files won't open**
- Solution: Try the XDMF file instead (outputs/data/xdmf/gravitational_field.xdmf)
"""
    
    with open(guide_path, "w") as f:
        f.write(guide_content)
    
    print(f"Quick start guide created: {guide_path}")

def create_advanced_script_info():
    info = """
Advanced ParaView Script Available!
====================================

For a more sophisticated multi-visualization setup, use:
  outputs/visualizations/paraview/paraview_advanced.py

This script creates:
  - Slice visualization (XY plane)
  - Contour/isosurface visualization
  - Vector field with glyphs
  - Streamline visualization

To use:
  1. Open ParaView
  2. Tools → Python Shell
  3. Run Script → Select 'outputs/visualizations/paraview/paraview_advanced.py'
  4. Toggle individual visualizations on/off in Pipeline Browser

Note: This script requires the output data files to exist in outputs/data/vtx/ or outputs/data/xdmf/.
"""
    print(info)

if __name__ == "__main__":
    print("=" * 70)
    print("PARAVIEW VISUALIZATION SETUP")
    print("=" * 70)
    
    create_paraview_python_script()
    print()
    create_paraview_state()
    print()
    create_quick_start_guide()
    
    print("\n" + "=" * 70)
    print("Setup files created successfully!")
    print("=" * 70)
    
    create_advanced_script_info()

