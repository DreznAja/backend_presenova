---
title: Presenova Face API
emoji: 🤖
colorFrom: green
colorTo: blue
sdk: docker
pinned: false
---

# Presenova Face Recognition API

Backend API untuk sistem absensi wajah Presenova.

## Endpoints

- `GET /health` — Cek status
- `POST /register-face` — Daftarkan wajah siswa
- `POST /recognize` — Kenali wajah dari kamera
- `POST /upload-scan-capture` — Upload foto saat scan
- `DELETE /delete-face/{id}` — Hapus data wajah

## Environment Variables

Set di HF Space Settings → Variables and secrets:

```
SUPABASE_URL=https://xxxx.supabase.co
SUPABASE_KEY=your_anon_key
```
