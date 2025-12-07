let
  pkgsBuiltin = import <nixpkgs>;
  
  unfreePkgs = [
    "libnpp"
    "libnvjitlink"
    "cudnn"
  ];

  # Choose nixpkgs version
  pkgs = pkgsBuiltin {
    config.allowUnfreePredicate = pkg: 
      let name = pkgs.lib.getName pkg; in
      builtins.elem name unfreePkgs || pkgs.lib.hasPrefix "cuda" name || pkgs.lib.hasPrefix "libcu" name;
  };
  pp = pkgs.python312Packages; # remove .venv after version change
in
pkgs.mkShell {
  buildInputs = [
    # Python
    pp.python
    pp.uv
    pp.venvShellHook
    pkgs.bashInteractive
    pkgs.ninja
  ] ++ pkgs.lib.optionals pkgs.stdenv.hostPlatform.isLinux [
    pkgs.libGLU
    pkgs.glib  # cv2: libgthread
    pkgs.glew
    pkgs.xorg.libX11  # imgui-bundle
    pkgs.xorg.libXext # imgui-bundle
    pkgs.cudatoolkit
    pkgs.cudaPackages.cudnn
    pkgs.cudaPackages.cuda_cudart
    pkgs.clang_19 # FastIsotropicMEdian
  ] ++ pkgs.lib.optionals pkgs.stdenv.hostPlatform.isDarwin [
    # github.com/NixOS/nixpkgs/blob/25.05/pkgs/development/python-modules/torchvision/default.nix#L64
    pkgs.apple-sdk_15
    (pkgs.darwinMinVersionHook "15.0")
  ];
  
  venvDir = "./.venv";
  
  postVenvCreation = ''
    uv pip install -e .
    uv pip install numpy pillow matplotlib pillow-avif-plugin
    uv pip uninstall ninja  # use nixpkgs version
  '';
  
  #LD_LIBRARY_PATH = pkgs.lib.makeLibraryPath (with pkgs; [
  #  wayland
  #]);
}
