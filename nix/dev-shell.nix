# Imported by flake.nix and shell.nix to provide the development environment.
{ pkgs, pythonPackages, packageSet }:

let
  python = pythonPackages.python.withPackages (_: [
    packageSet.pyviewer
    pythonPackages.matplotlib
    pythonPackages.pillow
    pythonPackages.torch
  ]);
in
pkgs.mkShell {
  packages = [
    python
    pkgs.bashInteractive
    pkgs.ninja
  ] ++ pkgs.lib.optionals pkgs.stdenv.hostPlatform.isLinux [
    pkgs.libglvnd
    pkgs.libGLU
    pkgs.glib
    pkgs.glew
    pkgs.libx11
    pkgs.libxext
    pkgs.cudatoolkit
    pkgs.cudaPackages.cudnn
    pkgs.cudaPackages.cuda_cudart
  ] ++ pkgs.lib.optionals pkgs.stdenv.hostPlatform.isDarwin [
    pkgs.apple-sdk_15
    (pkgs.darwinMinVersionHook "15.0")
  ];

  LD_LIBRARY_PATH = pkgs.lib.makeLibraryPath (with pkgs; [
    stdenv.cc.cc
    libjpeg
  ] ++ lib.optionals stdenv.hostPlatform.isLinux [
    libglvnd
    libx11
    libxext
    wayland
    libGL
    libxkbcommon
    cudatoolkit
    libz
    glib
  ]);
}
