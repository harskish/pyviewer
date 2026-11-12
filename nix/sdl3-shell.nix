# Shared SDL interop dependencies/environment for development and consumer shells.
{ pkgs, gpuInterop ? pkgs.stdenv.hostPlatform.isLinux }:

let
  # SDL 3.4.14 fixed Cocoa trackpad precision (scrollingDelta vs legacy delta).
  # Use an unmodified stable release when the caller's nixpkgs is older.
  stockSDL = if pkgs.stdenv.hostPlatform.isDarwin && pkgs.lib.versionOlder pkgs.sdl3.version "3.4.14"
    then pkgs.sdl3.overrideAttrs (_old: {
      version = "3.4.20";
      src = pkgs.fetchurl {
        url = "https://www.libsdl.org/release/SDL3-3.4.20.tar.gz";
        sha256 = "07sqz0zgp99ilwr7f0sqvzmrg574cbmg42khg8i5g5gdbgshdfzk";
      };
    }) else pkgs.sdl3;
  sdl3 = if gpuInterop then import ./sdl3-interop.nix { inherit pkgs; } else stockSDL;
in {
  packages = [ sdl3 pkgs.ninja ];
  env = {
    # Stock SDL on macOS; patched SDL_GPU bridge for Linux CUDA/Vulkan.
    # Set gpuInterop=true for GPU-backed SDL_Renderer interop on macOS.
    SDL_BINARY_PATH = "${pkgs.lib.getLib sdl3}/lib";
    SDL_DOWNLOAD_BINARIES = "0";
    SDL_DISABLE_METADATA = "1";
    SDL_DOC_GENERATOR = "0";
    SDL_CHECK_BINARY_VERSION = "0";
    SDL_CHECK_VERSION = "0";
  };
}
