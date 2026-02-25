// Story Printer — Display Stand (Kid-Friendly Booth)
// Houses: 7" HDMI Display (angled for kids), mic clip area
// Designed for Bambu Labs P1S (256x256x256mm build volume)
//
// Print settings: 0.2mm layer height, 15% infill, tree supports
// Material: PLA or PETG
// Print orientation: lay flat on the back for minimal supports

/* [Display Dimensions (mm)] */
// 7" HDMI display (common Raspberry Pi / Jetson display)
display_w = 170;    // screen width
display_h = 110;    // screen height
display_t = 10;     // screen thickness (including PCB)

/* [Stand Parameters] */
wall = 3;
clearance = 1.5;
corner_r = 5;

// Viewing angle (tilted back for kids to see while standing)
tilt_angle = 25;    // degrees from vertical

// Bezel lip to hold display in
lip = 5;            // how much the frame overlaps the screen edge
bezel_w = 8;        // frame width around the screen

// Stand base dimensions
base_depth = 100;   // how far back the stand extends on the table
base_height = 8;    // thickness of the base plate

/* [Cable Routing] */
cable_slot_w = 30;  // slot in the back for HDMI + USB cables
cable_slot_h = 15;

/* [Mic Holder] */
// Small clip on top to rest the Rode TX transmitter or a table mic
mic_clip_w = 50;
mic_clip_d = 50;
mic_clip_h = 25;
mic_clip_inner_w = 46; // Rode TX is 44mm wide
mic_clip_inner_d = 47; // Rode TX is 45.3mm deep

/* [Computed] */
// Frame outer dimensions
frame_w = display_w + bezel_w * 2 + wall * 2;
frame_h = display_h + bezel_w * 2 + wall * 2;
frame_t = display_t + clearance + wall;

// Stand total height when tilted
stand_height = frame_h * cos(tilt_angle) + frame_t * sin(tilt_angle) + base_height;
stand_depth_total = frame_h * sin(tilt_angle) + frame_t * cos(tilt_angle) + base_depth * 0.3;

echo(str("Frame dimensions: ", frame_w, " x ", frame_h, " x ", frame_t, " mm"));
echo(str("Stand height: ~", stand_height, " mm"));
echo(str("Fits P1S plate: ", frame_w <= 256 && frame_h <= 256 ? "YES" : "SPLIT PRINT"));

// -----------------------------------------------------------------------
// Modules
// -----------------------------------------------------------------------
module rounded_rect(w, h, r) {
    hull() {
        translate([r, r, 0]) circle(r=r, $fn=30);
        translate([w-r, r, 0]) circle(r=r, $fn=30);
        translate([r, h-r, 0]) circle(r=r, $fn=30);
        translate([w-r, h-r, 0]) circle(r=r, $fn=30);
    }
}

module rounded_box_3d(size, r) {
    hull() {
        for (x = [r, size[0]-r])
            for (y = [r, size[1]-r])
                translate([x, y, 0])
                    cylinder(r=r, h=size[2], $fn=30);
    }
}

// -----------------------------------------------------------------------
// Display frame (the part that holds the screen)
// -----------------------------------------------------------------------
module display_frame() {
    difference() {
        // Outer frame
        rounded_box_3d([frame_w, frame_h, frame_t], corner_r);

        // Screen pocket (recessed from the front)
        translate([bezel_w + wall, bezel_w + wall, wall])
            cube([display_w + clearance * 2, display_h + clearance * 2, display_t + clearance + 1]);

        // Viewing window (cut through the front face)
        translate([bezel_w + wall + lip, bezel_w + wall + lip, -1])
            cube([display_w - lip * 2, display_h - lip * 2, wall + 2]);

        // Cable exit slot (bottom edge of frame, centered)
        translate([frame_w/2 - cable_slot_w/2, -1, wall])
            cube([cable_slot_w, bezel_w + wall + 2, cable_slot_h]);

        // Cable exit slot (back, for HDMI ribbon)
        translate([frame_w/2 - cable_slot_w/2, frame_h - wall - 1, wall])
            cube([cable_slot_w, wall + 2, cable_slot_h]);
    }
}

// -----------------------------------------------------------------------
// Support struts (triangular, connect frame to base)
// -----------------------------------------------------------------------
module support_strut(height, depth, thickness) {
    linear_extrude(height=thickness)
        polygon(points=[
            [0, 0],
            [0, depth],
            [height, 0]
        ]);
}

// -----------------------------------------------------------------------
// Base plate with angled frame
// -----------------------------------------------------------------------
module display_stand() {
    // Base plate (sits flat on table)
    translate([0, 0, 0])
        rounded_box_3d([frame_w, base_depth, base_height], corner_r);

    // Frame, tilted back
    translate([0, base_height * 0.5, base_height])
        rotate([90 - tilt_angle, 0, 0])
            display_frame();

    // Left support strut
    strut_h = frame_h * cos(tilt_angle) * 0.7;
    strut_d = frame_h * sin(tilt_angle) * 0.8;
    translate([wall, base_depth * 0.3, base_height])
        rotate([0, 0, 0])
            support_strut(strut_h, strut_d, wall * 2);

    // Right support strut
    translate([frame_w - wall * 3, base_depth * 0.3, base_height])
        rotate([0, 0, 0])
            support_strut(strut_h, strut_d, wall * 2);
}

// -----------------------------------------------------------------------
// Mic holder clip (sits on top of the frame or beside it)
// -----------------------------------------------------------------------
module mic_clip() {
    difference() {
        // Outer box
        rounded_box_3d([mic_clip_w, mic_clip_d, mic_clip_h], 3);

        // Inner pocket for the Rode TX
        translate([
            (mic_clip_w - mic_clip_inner_w) / 2,
            (mic_clip_d - mic_clip_inner_d) / 2,
            wall
        ])
            cube([mic_clip_inner_w, mic_clip_inner_d, mic_clip_h]);
    }
}

// -----------------------------------------------------------------------
// Render
// -----------------------------------------------------------------------

// Stand with display
display_stand();

// Mic clip (positioned to the side — print separately)
translate([frame_w + 20, 0, 0])
    mic_clip();

// Ghost display for fit checking
%translate([bezel_w + wall, base_height * 0.5, base_height])
    rotate([90 - tilt_angle, 0, 0])
        translate([bezel_w + wall, bezel_w + wall, wall])
            color("Black", 0.3) cube([display_w, display_h, display_t]);
