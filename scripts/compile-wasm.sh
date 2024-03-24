#!/usr/bin/env bash

# Set default build type to 'dev' if not specified
BUILD_TYPE=${1:-dev}

# Common flags
COMMON_FLAGS="--target web --out-dir ./dist"

# Development build flags
DEV_FLAGS="--dev"

# Production build flags
PROD_FLAGS="--release"

# Select flags based on build type
if [ "$BUILD_TYPE" = "dev" ]; then
    FLAGS="$COMMON_FLAGS $DEV_FLAGS"
    echo "Building Development Version"
else
    FLAGS="$COMMON_FLAGS $PROD_FLAGS"
    echo "Building Production Version"
fi

# Web

# build ES6
echo "Building ES6"
wasm-pack build $FLAGS

# Remove .gitignore file from dist
rm ./dist/.gitignore

# check if terser is installed
if ! command -v terser &> /dev/null
then
	echo "terser could not be found, installing terser"
	npm install terser
fi

# minify the js file
echo "Minifying JS"
npx terser ./dist/simplify_rs.js --compress --output ./dist/simplify_rs.js
