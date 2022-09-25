# Overview
The goal is to predict silkscreen placement using a graph neural network. This is still in the testing/training phase. I can currently get the loss to decrease by several orders of magnitude, but it's not yet sufficient for accurate placement predictions. I welcome any contributions to attempt to solve this problem and believe solving silk placement would be a good proof of concept that component placement is solvable with this same architecture. Also, if anyone knows of altium or cadence projects that could be used in training please let me know.

# Getting Started
- Export data using altium/cadence script or use the example.csv file in the data/train/ directory
- Add data to projects' data/training/ directory
- Add data to projects' data/test/ directory
- Run training notebook (check data preprocessing)
- Once training, if you have a dataset in data/test/ the model will predict and export to data/predictions/predictions.csv and can be imported w/ the altium import script.
- Check progress in Logs notebook

# Data Collection
*Currently Supported PCB Tools: Altium & Cadence (Cadence only has export data script, not import script, and has not been as thoroughly vetted as altium data. I am less familiar w/ cadence)
- Use the export scripts in the scripts directory (Open pcb--> Run export)
- This project only uses the .PCBDoc [Altium] or .brd [Altium] file, so the full project does not need to be opened.

## Downloadable Projects
### Altium
- BeagleBone Black
- https://resources.altium.com/p/open-source-hardware-projects-in-altium-designer
- https://www.imx6rex.com (No Silkscreen Designators)
### Cadence
- Unknown

# Training
- Uses pytorch geometric to create a heterogeneous graph (different node types) of silkscreen, component, track, & arc nodes
- Predicts x, y, & rotation of the silkscreen designator (would like to predict size in the future)

# Graph Network Architecture
- Heterogeneous graph
- Inputs both categorical & numerical features
- Set the gnn type, number of layers, etc. in model.py

### Data Description:

| Column     | Type | Description |
| ----------- | ----------- | ----------- |
| Tool      | N/A     | altium/cadence|
| Type      | N/A       | Features extracted from type Silk or Pin (x/y/L/R/T/B). If Pin Type, features may be extracted from parent Component |
| Designator      | Component       | Parent Component Designator |
| x/y      | Component or Silk       | Origin Coords |
| L/R/T/B      | Component or Silk       | Bounding Rectangle (x1,x2,y1,y2 for tracks/arcs) |
| Rotation      | Component or Silk       | Degrees |
| Layer      | Component      | Top or Bottom |
| Info      | Type Specific      | Delimited list of extra info specific to Type |
| Board      | N/A     | This is not imported data. This is an index of each board/dataset |

### Info Column Specific Data

| Tool  | Info Column     | Type | Description |
| ----- | ----------- | ----------- | ----------- |
| Altium | PinName      | Pin     | Pin Name |
| Cadence | PinNumber      | Pin     | Pin Number |
| Cadence | PadstackName      | Pin     | Padstack Name |
| Cadence | PinRotation      | Pin     | Rotation in Degrees |
| Cadence | PinRelativeRotation      | Pin     | Rotation in Degrees Relative to Component |
| Cadence | IsThrough      | Pin     | Is a Through Hole pin |
| Both | NetName      | Pin     | Pin's Net Name |
| Both | PinX/Y      | Pin     | Pin's Origin Coords |
| Both | Width      | Track/Arc     | Track/Arc Width |
| Altium | Length      | Track     | Track Length |
| Both | Radius      | Arc     | Radius in mils |
| Altium | StartAngle      | Arc     | Start Angle |
| Altium | EndAngle      | Arc     | End Angle |
| Cadence | IsCircle      | Arc     | CIRCLE/UNCLOSED_ARC |
| Cadence | IsClockwise      | Arc     | TRUE/FALSE |
| Both | InComponent      | Track/Arc     | Object Part of Component Footprint (0=False,-1=True) |
| Cadence | LineType      | Track     | vertical/horizontal/odd |
| Cadence | Justify      | Des     | Text Justification (CENTER,) |
| Cadence | IsMirror      | Des     | Text is mirrored (TRUE/FALSE) |

Notes: 
- units in mils
- Pin Type varies for each row (Pin/Net/PinX/PinY), but Component data is the duplicate info across these rows (x/y/L/R/T/B/Rotation/Layer).

# References
- Grateful to the developers of the many software packages used throughout this project including, but not limited to, numpy, pytorch, pygeometric, pandas.
- https://github.com/pyg-team/pytorch_geometric
- https://pytorch-geometric.readthedocs.io/en/latest/notes/introduction.html
- https://colab.research.google.com/drive/14OvFnAXggxB8vM4e8vSURUp1TaKnovzX?usp=sharing#scrollTo=imGrKO5YH11-