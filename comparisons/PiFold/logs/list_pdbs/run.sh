#!/bin/bash
tar -tzvf mdcath_pdbs.tar.gz | head -20
echo ...; tar -tzf mdcath_pdbs.tar.gz | wc -l
f=$(tar -tzf mdcath_pdbs.tar.gz | grep -v '/$' | head -1); tar -xzf mdcath_pdbs.tar.gz "$f"; echo "== $f"; head -c 1500 "$f"; echo; grep -c "^ATOM" "$f"; awk '/^ATOM/{print substr($0,22,1)}' "$f" | sort | uniq -c
