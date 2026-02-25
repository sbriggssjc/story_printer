// Story Printer — Base Unit Case
// Houses: N80 Printer, Jetson Orin Nano, Anker 737, Rode Wireless GO II RX
// Designed for Bambu Labs P1S (256x256x256mm build volume)
//
// Print settings: 0.2mm layer height, 15% infill, tree supports
// Material: PLA or PETG

/* [Component Dimensions (mm)] */
// N80 Portable Thermal Printer
printer_w = 280;   // width (long axis)
printer_d = 84;    // depth
printer_h = 51;    // height

// Jetson Orin Nano (with connectors)
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

/* [Case Parameters] */
wall = 3;             // wall thickness
clearance = 2;        // extra space around each component
corner_r = 5;         // corner radius
vent_slot_w = 3;      // ventilation slot width
vent_slot_gap = 5;    // gap between vent slots

// Paper exit slot for the printer
paper_slot_w = 220;   // wide enough for A4/Letter paper
paper_slot_h = 8;     // paper thickness clearance

// Cable pass-through holes
cable_hole_d = 12;    // diameter for USB-C / HDMI cables

/* [Layout] */
// Components arranged side by side in the base:
//   [ Printer (full width) ]
//   [ Jetson | Anker | Rode ]

// Bottom row: electronics side by side
elec_total_w = (jetson_w + clearance) + (anker_w + clearance) + (rode_w + clearance) + wall * 2;

// Case interior width = max(printer width, electronics row)
interior_w = max(printer_w + clearance * 2, elec_total_w);
interior_d = (printer_d + clearance * 2) + wall + max(jetson_d, anker_d, rode_d) + clearance * 2;
interior_h = max(printer_h, max(jetson_h, anker_h)) + clearance * 2;

// Outer dimensions
case_w = interior_w + wall * 2;
case_d = interior_d + wall * 2;
case_h = interior_h + wall * 2;

echo(str("Case outer dimensions: ", case_w, " x ", case_d, " x ", case_h, " mm"));
echo(str("Fits on P1S build plate (256mm): width=", case_w <= 256 ? "YES" : "NO -- split print needed"));

/* [Computed positions] */
// Printer sits in the back (wide side)
printer_x = (interior_w - printer_w) / 2 + wall;
printer_y = wall + clearance;
printer_z = wall;

// Electronics row sits in front of the printer
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

// -----------------------------------------------------------------------
// Modules
// -----------------------------------------------------------------------
module rounded_box(size, r) {
    hull() {
        for (x = [r, size[0]-r])
            for (y = [r, size[1]-r])
                translate([x, y, 0])
                    cylinder(r=r, h=size[2], $fn=40);
    }
}

module vent_slots(length, height, slot_w, gap, count) {
    for (i = [0:count-1]) {
        translate([0, i * (slot_w + gap), 0])
            cube([wall + 1, slot_w, height]);
    }
}

module cable_hole(d) {
    rotate([0, 90, 0])
        cylinder(d=d, h=wall + 2, $fn=30);
}

// -----------------------------------------------------------------------
// Main case body
// -----------------------------------------------------------------------
module base_case() {
    difference() {
        // Outer shell
        rounded_box([case_w, case_d, case_h], corner_r);

        // Hollow interior
        translate([wall, wall, wall])
            rounded_box([interior_w, interior_d, interior_h + wall], corner_r - 1);

        // --- PRINTER PAPER EXIT SLOT (front face) ---
        translate([case_w/2 - paper_slot_w/2, -1, wall + printer_h - paper_slot_h])
            cube([paper_slot_w, wall + 2, paper_slot_h]);

        // --- VENTILATION: Jetson side (left wall) ---
        num_vents = 5;
        translate([-0.5, jetson_y + 10, jetson_z + 5])
            vent_slots(wall + 1, jetson_h - 10, vent_slot_w, vent_slot_gap, num_vents);

        // --- VENTILATION: Jetson top (open cutout above Jetson) ---
        translate([jetson_x + 5, jetson_y + 5, case_h - wall - 0.5])
            cube([jetson_w - 10, jetson_d - 10, wall + 1]);

        // --- CABLE PASS-THROUGHS ---
        // Jetson USB/HDMI (left side)
        translate([-0.5, jetson_y + jetson_d/2, jetson_z + jetson_h/2])
            cable_hole(cable_hole_d);

        // Jetson power barrel jack (left side, lower)
        translate([-0.5, jetson_y + jetson_d/2 + 20, jetson_z + 10])
            cable_hole(cable_hole_d);

        // Anker USB-C charging (right side)
        translate([case_w - wall - 0.5, anker_y + anker_d/2, anker_z + anker_h/2])
            cable_hole(cable_hole_d);

        // Rode USB-C (right side)
        translate([case_w - wall - 0.5, rode_y + rode_d/2, rode_z + rode_h/2])
            cable_hole(cable_hole_d);
    }

    // --- INTERNAL DIVIDER WALLS ---
    // Divider between printer bay and electronics bay
    translate([wall, printer_y + printer_d + clearance * 2, wall])
        cube([interior_w, wall, interior_h * 0.6]);

    // Divider between Jetson and Anker
    translate([anker_x - wall/2, elec_y, wall])
        cube([wall, max(jetson_d, anker_d) + clearance, interior_h * 0.5]);

    // Divider between Anker and Rode
    translate([rode_x - wall/2, elec_y, wall])
        cube([wall, max(anker_d, rode_d) + clearance, interior_h * 0.5]);
}

// -----------------------------------------------------------------------
// Render
// -----------------------------------------------------------------------
base_case();

// Component visualization (for fit checking — comment out before export)
%translate([printer_x, printer_y, printer_z])
    color("DimGray", 0.5) cube([printer_w, printer_d, printer_h]);

%translate([jetson_x, jetson_y, jetson_z])
    color("Green", 0.5) cube([jetson_w, jetson_d, jetson_h]);

%translate([anker_x, anker_y, anker_z])
    color("DarkBlue", 0.5) cube([anker_w, anker_d, anker_h]);

%translate([rode_x, rode_y, rode_z])
    color("Red", 0.5) cube([rode_w, rode_d, rode_h]);
