# Imported by flake.nix and shell.nix to provide the development environment.
{ pkgs, pythonPackages, packageSet }:

let
  python = pythonPackages.python.withPackages (_: [
    packageSet.pyviewer
    pythonPackages.matplotlib
    pythonPackages.pillow
    pythonPackages.pysdl3
    pythonPackages.torch
  ]);
  pythonVersion = pkgs.lib.versions.majorMinor pythonPackages.python.version;
in
pkgs.mkShell {
  # Python's withPackages environment does not run PySDL3's setup hook.
  # Use its Nix-provided libraries instead of downloading into site-packages.
  SDL_DISABLE_METADATA = "1";
  SDL_DOWNLOAD_BINARIES = "0";
  # Runtime doc generation writes __doc__.py into the read-only Nix store.
  SDL_DOC_GENERATOR = "0";
  SDL_CHECK_BINARY_VERSION = "0";
  SDL_CHECK_VERSION = "0";

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
    zlib
    glib
  ]);

  shellHook = ''
    pyviewer_root="$(git rev-parse --show-toplevel 2>/dev/null || pwd)"
    if [ -f "$pyviewer_root/pyproject.toml" ] && [ -f "$pyviewer_root/nix/dev-shell.nix" ]; then
      export PYTHONPATH="$pyviewer_root''${PYTHONPATH:+:$PYTHONPATH}"
      mkdir -p "$pyviewer_root/.nix-python/bin"
      ln -sfn "${python}/bin/python" "$pyviewer_root/.nix-python/bin/python${pythonVersion}"
      ln -sfn "python${pythonVersion}" "$pyviewer_root/.nix-python/bin/python3"
      ln -sfn "python${pythonVersion}" "$pyviewer_root/.nix-python/bin/python"
    fi
  '';
}
