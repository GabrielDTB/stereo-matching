with import <nixpkgs> {};
let python =
  let
  packageOverrides = self:
  super: {
    opencv4 = super.opencv4.override {
      enableGtk2 = true;
      gtk2 = pkgs.gtk2;
      enableFfmpeg = true; #here is how to add ffmpeg and other compilation flags
      };
  };
  in
    pkgs.python311.override {inherit packageOverrides; self = python;};

in

mkShell {
  buildInputs = with python311.pkgs; [
    python311
    python.pkgs.opencv4
    numpy
    matplotlib
    pkgconfig
    gtk3
    gtk2
    gtk2-x11
  ];
}
