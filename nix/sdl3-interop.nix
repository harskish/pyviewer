{ pkgs }:

assert pkgs.lib.assertMsg (pkgs.sdl3.version == "3.4.10")
  "pyviewer native interop patch requires SDL 3.4.10; update and validate the patch before changing SDL versions";
pkgs.sdl3.overrideAttrs (old: {
  patches = (old.patches or []) ++ [ ../pyviewer/custom_ops/sdl3-interop.patch ];
})
