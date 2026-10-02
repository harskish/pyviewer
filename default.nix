# Entry point for nix-build or `import ./.`; exposes the Python package set.
{ pkgs ? import <nixpkgs> {
    config.allowUnfreePredicate = pkg:
      (pkg.pname or "") == "pyviewer";
  }
, pythonPackages ? pkgs.python312Packages
}:

import ./nix/python-packages.nix {
  inherit pkgs pythonPackages;
}
