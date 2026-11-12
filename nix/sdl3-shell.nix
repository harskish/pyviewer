# Shared SDL interop dependencies/environment for development and consumer shells.
{ pkgs }:

let
  sdl3 = import ./sdl3-interop.nix { inherit pkgs; };
in {
  packages = [ sdl3 pkgs.ninja ];
  env = {
    # PySDL3 must use this patched library, without writing into site-packages.
    SDL_BINARY_PATH = "${pkgs.lib.getLib sdl3}/lib";
    SDL_DOWNLOAD_BINARIES = "0";
    SDL_DISABLE_METADATA = "1";
    SDL_DOC_GENERATOR = "0";
    SDL_CHECK_BINARY_VERSION = "0";
    SDL_CHECK_VERSION = "0";
  };
}
