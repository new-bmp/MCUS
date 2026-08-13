#!/usr/bin/env python3
from import_vendor_official import main

if __name__ == "__main__":
    import sys
    sys.argv.insert(1, "microchip")
    raise SystemExit(main())
