# Story Printer — 3D Printed Case

Two-piece kid-friendly booth for the invention convention.

## Components & Dimensions

| Component | W × D × H (mm) | Bay |
|---|---|---|
| N80 Thermal Printer | 280 × 84 × 51 | Base unit (back) |
| Jetson Orin Nano | 103 × 91 × 35 | Base unit (front-left) |
| Anker 737 (24K) | 156 × 55 × 50 | Base unit (front-center) |
| Rode Wireless GO II RX | 44 × 46 × 18 | Base unit (front-right) |
| 7" HDMI Display | 170 × 110 × 10 | Display stand |

## Pieces to Print

### 1. `base_unit.scad` — Main electronics enclosure
- Houses the printer, Jetson, Anker battery, and Rode receiver
- Paper exit slot in the front for printed stories
- Ventilation over the Jetson + cable pass-throughs
- Internal divider walls between compartments
- **NOTE**: The base unit is ~310mm wide (because the N80 printer is 280mm).
  The Bambu P1S build plate is 256mm. You'll need to either:
  - **Option A**: Print in two halves and glue/bolt together (split at the
    divider wall between printer bay and electronics bay)
  - **Option B**: Print diagonally (310mm fits on 256×256 diagonal = 362mm)
  - **Option C**: Scale down clearances or rotate the printer 90°

### 2. `display_stand.scad` — Angled monitor stand + mic clip
- 25° tilt angle (comfortable for kids standing at a table)
- Lip bezel holds the display in place
- Cable routing slots for HDMI and USB
- Separate mic clip piece (snap-fit tray for the Rode TX transmitter)
- Fits on the P1S build plate: ~204mm wide

### 3. Mic clip (included in display_stand.scad)
- Small tray that holds the Rode Wireless GO II transmitter
- Print separately, place on the table or attach to the stand

## Print Settings (Bambu P1S)

| Setting | Value |
|---|---|
| Layer height | 0.2mm |
| Infill | 15% (grid or gyroid) |
| Walls | 3 perimeters |
| Supports | Tree supports (for cable holes and overhangs) |
| Material | PLA or PETG |
| Plate adhesion | Brim (for the large base unit) |

## Assembly

1. Print base unit and display stand
2. Drop components into their bays (friction fit with 2mm clearance)
3. Route cables through the pass-through holes:
   - Jetson: USB-C power from Anker, HDMI to display, USB-A for WiFi dongle
   - Anker: USB-C charging cable out the side
   - Rode RX: 3.5mm audio to Jetson USB audio adapter (or USB-C)
   - Printer: USB-C from Jetson (or Bluetooth)
4. Slide the 7" display into the frame bezel
5. Place the Rode TX in the mic clip on the table
6. Connect keyboard via USB

## Customizing

Both `.scad` files have parametric dimensions at the top. To adjust:

1. Install [OpenSCAD](https://openscad.org/) (free)
2. Open the `.scad` file
3. Modify the dimension variables in the `[Component Dimensions]` section
4. Press F5 to preview, F6 to render
5. Export as STL: File → Export → STL
6. Import the STL into Bambu Studio and slice

If your components are slightly different sizes, just update the numbers
and re-export.
