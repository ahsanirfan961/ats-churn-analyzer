#!/bin/sh
set -eu

API_URL="${API_BASE_URL:-${VITE_API_BASE_URL:-http://localhost:8000}}"
ESCAPED_API_URL=$(printf '%s' "$API_URL" | sed 's/\\/\\\\/g; s/"/\\"/g')

cat > /usr/share/nginx/html/config.js <<EOF
window.__APP_CONFIG__ = { apiBaseUrl: "$ESCAPED_API_URL" };
EOF

exec nginx -g 'daemon off;'
