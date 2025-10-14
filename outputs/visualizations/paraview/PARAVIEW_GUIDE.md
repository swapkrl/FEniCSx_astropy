# ParaView Quick Start Guide

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
