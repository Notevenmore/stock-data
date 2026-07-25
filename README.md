# Stock Data Mining

Project ini merupakan project untuk mendukung pengumpulan data-data berita, data-data _orderbook_, dan data-data teknikal emiten

Project ini akan digunakan oleh **Stock-ML Prediction** untuk memprediksi kenaikan ataupun penurunan harga saham yang ada di emiten IHSG.

Project ini masih dalam pengembangan dan dapat dioptimalisasi data nya dengan menambahkan pengumpulan data berita International (saat ini, data berita sebatas berita dalam negeri)

## Installation Guide

### STOCK-DATA-APP-DOCUMENTATION

1. Inisialisasi virtual environtment untuk project terlebih dahulu

```bash
cd stock-data
python -m venv venv
```

2. Setelah itu, aktifkan virtual environtmentnya

```bash
source venv/bin/activate
```

3. Setelah mengaktifkan virtual environtment, install seluruh package yang ada di requirements.txt

```bash
pip install -r requirements.txt
```

4. Setelah melakukan inisialisasi virtual environtment, salin file .env.example dan atur data-data yang kosong (di semua folder (app, celery, ataupun ml-server) lakukan hal ini)

```bash
cd <some-project-directory>
cp .env.example .env
```

Folder App

```.env
MEDIA_KEYWORDS=
KEYWORD_NATIONAL=
KEYWORD_INTERNATIONAL=

START=
MONTHS=

CATEGORY=
LIST_MEDIA=
MEDIA_URL=
ORDERBOOK_URL=

DB_USERNAME=
DB_PASSWORD=
DB_NAME=
DB_PORT=
DB_HOST=
```

Folder Celery

```.env
CELERY_BROKER_URL=
CELERY_RESULT_BACKEND=
SCRAPPING_API_URL=
SCRAPPING_MINUTES=
SCRAPPING_HOUR=
```

5. Jika ingin menambahkan package baru, jalankan command berikut diterminal

```bash
pip install <new-package>
pip freeze > requirements.txt
```

6. Tahapan selanjutnya, inisialisasi database. dengan menggunakan kode berikut (jika pertama kali diload atau belum ada folder migrations)

```bash
cd app
flask --app main db init
flask --app main db migrate -m "<new migrate>"
flask --app main db upgrade
```

7. Jika ingin menambahkan data dummy di database, jalankan kode berikut diterminal

```bash
cd app
python3 -m database/seed.py
```

### STOCK-DATA-CELERY-DOCUMENTATION (For Realtime and Notify newest news, orderbook, and analyze)

8. Untuk mengaktifkan fitur realtime + scheduling data yang akan dimuat setiap hari (tepatnya di jam 12), lakukan tahapan selanjutnya

9. Aktifkan server redis (MacOS Only)

```bash
brew services start redis
```

Memastikan server redis telah berjalan

```bash
brew services list | grep redis
redis-cli ping
```

Untuk menonaktifkan:

```bash
brew services stop redis
```

Untuk restart:

```bash
brew services restart redis
```

10. Untuk melakukan pengujian fitur, jalankan

Fitur app

```bash
cd app
source ./test/test.sh
```

Fitur celery

```bash
cd celery
python -m test.workflow
```

11. Jalankan celery worker, beat dan server

```bash
   cd celery
   celery -A celery_app worker --loglevel=info
```

```bash
   cd celery
   celery -A celery_app beat --loglevel=info
```

```bash
   cd app
   python main.py
```

12. Jalankan ML Server

```bash
    cd ml-server
    python main.py
```

13. Untuk Modifikasi Model di Ml-Server, gunakan playground untuk menguji atau menambahkan model. hal ini dapat dilakukan dengan menyalin file .ipynb yang ada, lalu modifikasi file untuk menambah variasi model baru. Atau jika ingin memperbarui model yang sudah ada, lakukan perubahan di file .ipynb yang sudah ada.

14. Jika model yang dibangun sudah pas, salin kode yang berkaitan tersebut di class suatu model di project ml-server, sesuaikan untuk mendapatkan hasil yang optimal

## Arsitektur Sistem

```text
                       +----------------------+
                       |     PostgreSQL       |
                       +----------+-----------+
                                  ^
                                  |
                    +-------------+-------------+
                    |                           |
          +---------+---------+       +---------+---------+
          |     Flask API     |       | Celery Worker     |
          | REST / WebSocket  |       | Background Jobs   |
          +---------+---------+       +---------+---------+
                    ^                           ^
                    |                           |
                    |                    +------+------+
                    |                    |    Redis    |
                    |                    +------+------+
                    |                           ^
                    |                           |
              +-----+---------------------------+
              |
      +-------+--------+
      |  Client Apps   |
      +----------------+
      | Desktop (Qt)   |
      | Web            |
      | Android        |
      | iOS            |
      +----------------+
```

## Scheduler

```text
          Every 00:00
               │
               ▼
       Celery Beat
               │
               ▼
            Redis
               │
               ▼
        Celery Worker
               │
               ▼
       daily_update()
```

## Daily Workflow

```text
                daily_update
                     │
                  chord
                     │
      ┌──────────────┴──────────────┐
      │                             │
      ▼                             ▼
Orderbook                 News
                                │
                                ▼
                        Processed Data
      │                             │
      └──────────────┬──────────────┘
                     ▼
                  Stock
```

## API Flow

```text
Desktop
Web
Android
iOS
    │
    ▼
Flask REST API
    │
    ▼
Service Layer
    │
    ▼
Repository Layer
    │
    ▼
Database
```

Frontend: NextJS
Desktop: PySide6
Android: Kotlin
IOS: SwiftUI
