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

### 1. `base_unit.scad` — Main electronics enclosure (TWO halves)

The full base is ~321mm wide (the N80 printer is 280mm), which exceeds the
P1S build plate (256mm). The design splits it into **left and right halves**
(~161mm each) that connect with alignment pins and glue.

**How to export the two STL files:**

1. Open `base_unit.scad` in OpenSCAD
2. Change `render_half = "left"` on line 21
3. Press **F6** (render), then **File → Export as STL** → save as `base_left.stl`
4. Change `render_half = "right"`
5. Press **F6** again, export as `base_right.stl`
6. Set `render_half = "both"` to preview the assembled case (don't export this)

**Joining the two halves:**

1. Print both halves
2. The left half has 4 alignment pins; the right half has matching holes
3. Dry-fit the halves to check alignment
4. Apply super glue (or PLA cement) to the seam face
5. Press the pins into the holes and hold for 30 seconds
6. Optional: run a bead of super glue along the outside seam for extra strength

### 2. `display_stand.scad` — Angled monitor stand + mic clip

- 25° tilt angle (comfortable for kids standing at a table)
- Lip bezel holds the display in place
- Cable routing slots for HDMI and USB
- **Fits on the P1S plate**: ~192mm wide (no split needed)

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
| Plate adhesion | Brim (for the large base halves) |

## Assembly

1. Print left base half, right base half, display stand, and mic clip (4 prints)
2. Glue the two base halves together using the alignment pins
3. Drop components into their bays (friction fit with 2mm clearance):
   - Printer in the back bay
   - Jetson in the front-left bay
   - Anker battery in the front-center bay
   - Rode receiver in the front-right bay
4. Route cables through the pass-through holes:
   - Jetson: USB-C power from Anker, HDMI to display, USB-A for WiFi dongle
   - Anker: USB-C charging cable out the right side
   - Rode RX: 3.5mm audio to Jetson USB audio adapter (or USB-C)
   - Printer: USB-C from Jetson (or Bluetooth)
5. Slide the 7" display into the frame bezel on the stand
6. Place the Rode TX in the mic clip on the table
7. Connect keyboard via USB

## Customizing

Both `.scad` files have parametric dimensions at the top. To adjust:

1. Install [OpenSCAD](https://openscad.org/) (free)
2. Open the `.scad` file
3. Modify the dimension variables at the top
4. Press F5 to preview, F6 to render
5. Export as STL: File → Export → STL
6. Import the STL into Bambu Studio and slice

If your components are slightly different sizes, just update the numbers
and re-export. The split point, alignment pins, and internal dividers all
adjust automatically.
