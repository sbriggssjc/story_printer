// Story Printer — Base Unit Case (Two-Piece Split Design)
// Houses: N80 Printer, Jetson Orin Nano, Anker 737, Rode Wireless GO II RX
// Designed for Bambu Labs P1S (256 × 256 × 256mm build volume)
//
// The full case is ~321mm wide — too big for the P1S plate.
// Solution: split into LEFT and RIGHT halves (~161mm each).
// Halves connect with alignment pins + super glue or PLA cement.
//
// EXPORT INSTRUCTIONS:
//   1. Set render_half = "left"  → press F6 → File > Export as STL
//   2. Set render_half = "right" → press F6 → File > Export as STL
//   3. Print both halves
//   4. Press alignment pins into left half, apply glue to seam, press together
//
// Print settings: 0.2mm layer height, 15% infill, tree supports
// Material: PLA or PETG

/* =================================================================
   RENDER CONTROL — change this before each STL export
   ================================================================= */
render_half = "both";   // "left", "right", or "both" (preview only)

/* =================================================================
   Component Dimensions (mm)
   Measure your actual components and update if needed!
   ================================================================= */
// N80 Portable Thermal Printer
printer_w = 280;    // width (long axis)
printer_d = 84;     // depth
printer_h = 51;     // height

// Jetson Orin Nano (with carrier board)
jetson_w = 103;
jetson_d = 91;
jetson_h = 35;

// Anker 737 PowerCore 24K
anker_w = 156;
anker_d = 55;
anker_h = 50;

// Rode Wireless GO II Receiver
rode_w = 44;
rode_d = 46;
rode_h = 18;

/* =================================================================
   Case Parameters
   ================================================================= */
wall        = 3;        // wall thickness
clearance   = 2;        // space around each component
corner_r    = 5;        // rounded corner radius
vent_slot_w = 3;        // ventilation slot width
vent_slot_gap = 5;      // gap between vent slots

// Paper exit slot (front face, for printed stories)
paper_slot_w = 220;     // wide enough for thermal paper roll
paper_slot_h = 8;       // clearance for paper

// Cable pass-through holes
cable_hole_d = 12;      // diameter for USB-C / HDMI cables

/* =================================================================
   Split & Join Parameters
   ================================================================= */
pin_d       = 4;        // alignment pin diameter
pin_len     = 6;        // pin protrusion length
pin_tol     = 0.3;      // extra diameter for pin holes (print tolerance)
num_pins    = 4;        // pins along the seam

/* =================================================================
   Layout Computation
   ================================================================= */
// Bottom row: Jetson + Anker + Rode side by side
elec_total_w = (jetson_w + clearance) + (anker_w + clearance)
             + (rode_w + clearance) + wall * 2;

// Interior = whichever row is wider
interior_w = max(printer_w + clearance * 2, elec_total_w);
interior_d = (printer_d + clearance * 2) + wall
           + max(jetson_d, anker_d, rode_d) + clearance * 2;
interior_h = max(printer_h, max(jetson_h, anker_h)) + clearance * 2;

// Outer dimensions
case_w = interior_w + wall * 2;
case_d = interior_d + wall * 2;
case_h = interior_h + wall * 2;

// Split at the centerline
split_x = case_w / 2;

echo(str("Full case: ", case_w, " x ", case_d, " x ", case_h, " mm"));
echo(str("Each half: ~", split_x, " mm wide — fits P1S (256mm): ",
         split_x <= 256 ? "YES" : "NO"));

/* =================================================================
   Component Positions
   ================================================================= */
// Printer in the back row (centered)
printer_x = (interior_w - printer_w) / 2 + wall;
printer_y = wall + clearance;
printer_z = wall;

// Electronics row in front of the printer
elec_y = printer_y + printer_d + clearance * 2 + wall;

// Jetson on the left
jetson_x = wall + clearance;
jetson_y = elec_y;
jetson_z = wall;

// Anker in the middle
anker_x = jetson_x + jetson_w + clearance + wall;
anker_y = elec_y;
anker_z = wall;

// Rode on the right
rode_x = anker_x + anker_w + clearance + wall;
rode_y = elec_y;
rode_z = wall;

/* =================================================================
   Pin Positions (along seam at x = split_x)
   ================================================================= */
function pin_y(i) = wall + 20
    + i * ((case_d - wall * 2 - 40) / (num_pins - 1));
pin_z = case_h / 2;

/* =================================================================
   Utility Modules
   ================================================================= */
module rounded_box(size, r) {
    hull() {
        for (x = [r, size[0] - r])
            for (y = [r, size[1] - r])
                translate([x, y, 0])
                    cylinder(r = r, h = size[2], $fn = 40);
    }
}

module vent_slots(length, height, sw, gap, count) {
    for (i = [0 : count - 1])
        translate([0, i * (sw + gap), 0])
            cube([wall + 1, sw, height]);
}

module cable_hole(d) {
    rotate([0, 90, 0])
        cylinder(d = d, h = wall + 2, $fn = 30);
}

/* =================================================================
   Full Case (un-split) — used as the cutting source
   ================================================================= */
module full_case() {
    difference() {
        // Outer shell
        rounded_box([case_w, case_d, case_h], corner_r);

        // Hollow interior
        translate([wall, wall, wall])
            rounded_box([interior_w, interior_d, interior_h + wall],
                        corner_r - 1);

        // --- Paper exit slot (front face) ---
        translate([case_w / 2 - paper_slot_w / 2, -1,
                   wall + printer_h - paper_slot_h])
            cube([paper_slot_w, wall + 2, paper_slot_h]);

        // --- Ventilation: Jetson side (left wall) ---
        translate([-0.5, jetson_y + 10, jetson_z + 5])
            vent_slots(wall + 1, jetson_h - 10,
                       vent_slot_w, vent_slot_gap, 5);

        // --- Ventilation: Jetson top cutout ---
        translate([jetson_x + 5, jetson_y + 5, case_h - wall - 0.5])
            cube([jetson_w - 10, jetson_d - 10, wall + 1]);

        // --- Cable pass-throughs ---
        // Jetson USB / HDMI (left wall)
        translate([-0.5, jetson_y + jetson_d / 2,
                   jetson_z + jetson_h / 2])
            cable_hole(cable_hole_d);

        // Jetson power barrel jack (left wall, lower)
        translate([-0.5, jetson_y + jetson_d / 2 + 20, jetson_z + 10])
            cable_hole(cable_hole_d);

        // Anker USB-C charging (right wall)
        translate([case_w - wall - 0.5, anker_y + anker_d / 2,
                   anker_z + anker_h / 2])
            cable_hole(cable_hole_d);

        // Rode USB-C (right wall)
        translate([case_w - wall - 0.5, rode_y + rode_d / 2,
                   rode_z + rode_h / 2])
            cable_hole(cable_hole_d);
    }

    // --- Internal divider walls ---
    // Divider between printer bay and electronics bay
    translate([wall, printer_y + printer_d + clearance * 2, wall])
        cube([interior_w, wall, interior_h * 0.6]);

    // Divider between Jetson and Anker
    translate([anker_x - wall / 2, elec_y, wall])
        cube([wall, max(jetson_d, anker_d) + clearance,
              interior_h * 0.5]);

    // Divider between Anker and Rode
    translate([rode_x - wall / 2, elec_y, wall])
        cube([wall, max(anker_d, rode_d) + clearance,
              interior_h * 0.5]);
}

/* =================================================================
   LEFT HALF — prints seam-side up or flat
   ================================================================= */
module left_half() {
    difference() {
        union() {
            // Cut the full case at the split line
            intersection() {
                full_case();
                translate([-1, -1, -1])
                    cube([split_x + 1, case_d + 2, case_h + 2]);
            }

            // Alignment pins (protrude rightward from seam face)
            for (i = [0 : num_pins - 1]) {
                translate([split_x, pin_y(i), pin_z])
                    rotate([0, 90, 0])
                        cylinder(d = pin_d, h = pin_len, $fn = 24);
            }

            // Seam reinforcement strip (thickens the wall at the cut)
            translate([split_x - wall, wall + 1, wall + 1])
                cube([wall, interior_d - 2, interior_h - 2]);
        }
        // (no subtractions needed for left half)
    }
}

/* =================================================================
   RIGHT HALF — prints seam-side up or flat
   ================================================================= */
module right_half() {
    difference() {
        union() {
            // Cut the full case at the split line
            intersection() {
                full_case();
                translate([split_x, -1, -1])
                    cube([case_w - split_x + 1, case_d + 2,
                          case_h + 2]);
            }

            // Seam reinforcement strip
            translate([split_x, wall + 1, wall + 1])
                cube([wall, interior_d - 2, interior_h - 2]);
        }

        // Alignment pin holes (matching pins from left half)
        for (i = [0 : num_pins - 1]) {
            translate([split_x - 0.5, pin_y(i), pin_z])
                rotate([0, 90, 0])
                    cylinder(d = pin_d + pin_tol,
                             h = pin_len + 1, $fn = 24);
        }
    }
}

/* =================================================================
   Render
   ================================================================= */
if (render_half == "left") {
    left_half();
} else if (render_half == "right") {
    // Shift to origin so the slicer centers it on the plate
    translate([-split_x, 0, 0])
        right_half();
} else {
    // "both" — assembled preview with a 2mm gap at the seam
    color("SteelBlue", 0.9)  left_half();
    color("Coral", 0.9)      translate([2, 0, 0]) right_half();

    // Ghost components for fit checking
    %translate([printer_x, printer_y, printer_z])
        color("DimGray", 0.5) cube([printer_w, printer_d, printer_h]);
    %translate([jetson_x, jetson_y, jetson_z])
        color("Green", 0.5) cube([jetson_w, jetson_d, jetson_h]);
    %translate([anker_x, anker_y, anker_z])
        color("DarkBlue", 0.5) cube([anker_w, anker_d, anker_h]);
    %translate([rode_x, rode_y, rode_z])
        color("Red", 0.5) cube([rode_w, rode_d, rode_h]);
}
