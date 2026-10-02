# Entry point for nix build and nix develop; exposes packages and dev shells.
{
  description = "Interactive Python viewers";

  inputs.nixpkgs.url = "github:NixOS/nixpkgs/nixos-26.05";

  outputs = { nixpkgs, ... }:
    let
      systems = [
        "x86_64-linux"
        "x86_64-darwin"
        "aarch64-darwin"
      ];
      forAllSystems = nixpkgs.lib.genAttrs systems;
      mkPkgs = system: cudaSupport: import nixpkgs {
        inherit system;
        config.cudaSupport = cudaSupport;
        config.allowUnfree = cudaSupport;
        config.allowUnfreePredicate = pkg:
          let name = nixpkgs.lib.getName pkg; in
          builtins.elem name [
            "pyviewer"
            "libnpp"
            "libnvjitlink"
            "cudnn"
          ]
          || nixpkgs.lib.hasPrefix "cuda" name
          || nixpkgs.lib.hasPrefix "libcu" name;
      };
      mkPackageSet = pkgs: pythonPackages: import ./nix/python-packages.nix {
        inherit pkgs pythonPackages;
      };
      pythonPackageSets = pkgs: {
        "311" = pkgs.python311Packages;
        "312" = pkgs.python312Packages;
        "313" = pkgs.python313Packages;
        "314" = pkgs.python314Packages;
      };
    in
    {
      packages = forAllSystems (system:
        let
          pkgs = mkPkgs system false;
          packageSets = nixpkgs.lib.mapAttrs
            (_: pythonPackages: mkPackageSet pkgs pythonPackages)
            (pythonPackageSets pkgs);
          defaultPackageSet = packageSets."312";
        in
        {
          default = defaultPackageSet.pyviewer;
          inherit (defaultPackageSet) pyviewer;
          imgui-bundle = defaultPackageSet.imgui-bundle;
          light-process = defaultPackageSet.light-process;
          python-wayland = defaultPackageSet.python-wayland;
          pyviewer311 = packageSets."311".pyviewer;
          pyviewer312 = packageSets."312".pyviewer;
          pyviewer313 = packageSets."313".pyviewer;
          pyviewer314 = packageSets."314".pyviewer;
        });

      devShells = forAllSystems (system:
        let
          pkgs = mkPkgs system (system == "x86_64-linux");
          shells = nixpkgs.lib.mapAttrs'
            (version: pythonPackages:
              let
                packageSet = mkPackageSet pkgs pythonPackages;
                shell = import ./nix/dev-shell.nix {
                  inherit pkgs pythonPackages packageSet;
                };
              in
              nixpkgs.lib.nameValuePair "python${version}" shell)
            (pythonPackageSets pkgs);
        in
        shells // { default = shells.python313; });
    };
}
