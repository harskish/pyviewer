# Entry point for nix-shell (and direnv's `use nix`); imports the shared dev shell.
{ pythonVersion ? "3.13" }:

let
  pkgsBuiltin = import <nixpkgs>;

  unfreePkgs = [
    "pyviewer"
    "libnpp"
    "libnvjitlink"
    "cudnn"
  ];

  pkgs = pkgsBuiltin {
    config.cudaSupport = builtins.currentSystem == "x86_64-linux";
    config.allowUnfree = builtins.currentSystem == "x86_64-linux";
    config.allowUnfreePredicate = pkg:
      let name = pkgs.lib.getName pkg; in
      builtins.elem name unfreePkgs
      || pkgs.lib.hasPrefix "cuda" name
      || pkgs.lib.hasPrefix "libcu" name;
  };
  pythonPackageSets = {
    "3.11" = pkgs.python311Packages;
    "3.12" = pkgs.python312Packages;
    "3.13" = pkgs.python313Packages;
    "3.14" = pkgs.python314Packages;
  };
  pp = pythonPackageSets.${pythonVersion} or
    (throw "pyviewer: unsupported Python version ${pythonVersion}; expected 3.11, 3.12, 3.13, or 3.14");
  localPackages = import ./nix/python-packages.nix {
    inherit pkgs;
    pythonPackages = pp;
  };
in
import ./nix/dev-shell.nix {
  inherit pkgs;
  pythonPackages = pp;
  packageSet = localPackages;
}
