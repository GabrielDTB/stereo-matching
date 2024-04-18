with import <nixpkgs> {};
mkShell {
  buildInputs = with python311.pkgs; [
    python311
    opencv4
    numpy
    matplotlib
  ];
}
