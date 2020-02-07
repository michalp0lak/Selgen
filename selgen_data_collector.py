#!/usr/bin/python
# -*- coding: utf-8 -*-

# - nastavit ssh RSA key z jednoho kompu na druhý
# - bloknout striktni kontrolu ukládání host keys (ssh.conf)
# - nastavit MAC adresu v dnsmasq serveru a přidělovat jedinou IP


import paramiko
import time
import os

camset_file = '/home/tester/Desktop/nastaveni_kamery.set'
local_datadir = '/home/tester/Desktop/data/'
rpi_datadir = '/home/pi/data/'
raspi_addr = '10.42.0.10'
user = 'pi'
remid = user + '@' + raspi_addr + ':'


#luser = os.system('whoami')
#lhost = os.system('hostname')

# parametry fotografie - soubor nastaveni_kamery.set
with open(camset_file, 'r') as f:
    sett = f.readlines()

exp = sett[0][(sett[0].index('=')+1):sett[0].index('u')]
awb = sett[1][(sett[1].index('=')+1):]
awb = awb.rstrip()

# definice vzdalenych prikazu
rpistill = 'raspistill -v -n -md 2 -awb auto -ss 10000 -o '
lrsync = 'rsync -vat ' + remid + rpi_datadir + '* ' + local_datadir

print('Systemovy cas: ' + time.asctime(time.localtime(time.time())))


if not os.path.isdir(local_datadir):
    os.mkdir(local_datadir)

# připojení
client = paramiko.SSHClient()
client.load_system_host_keys()
client.set_missing_host_key_policy(paramiko.AutoAddPolicy())

# dodelat
# client.set_log_channel()
# uziti privatniho klice:
# k = paramiko.RSAKey.from_private_key_file('/cesta/ke/klici/')

try:    # vyladit!
    client.connect(raspi_addr, port=22, username=user, password='raspberry')
except paramiko.ssh_exception.SSHException:
    exit()
    print('Chyba SSH pripojeni.')


if client.get_transport().is_active():
    print('Vstup do OK smycky pro foceni, jsme pripojeni.')
    while client.get_transport().is_active():

        var = input("Read barcode of tray and confirm decoded data with ENTER button: ")
        print('String: ' + var + "  will be used in filename of image.")

        # cekam na imput
        x = raw_input('Stiskněte enter pro sejmutí fotografie.')
            
        # jmeno fotografie s timestampem
        timestamp = time.strftime('datum_%Y-%m-%d_cas_%H-%M-%S')
        picname = var + '_' + timestamp + '.png'

        # vypis jmena fotky pro uzivatele
        print("Jmeno fotky: " picname)

        # command pro vyfocení
        takepic_sh = rpistill + rpi_datadir + picname

        # kontrola prubehu procesu
        # nacist status a pockat na dokonceni foceni!
        print('Probíhá focení..')
        i,o,e = client.exec_command(takepic_sh)

        # cekani na exit status
        exit_st1 = o.channel.recv_exit_status()
        if exit_st1 == 0:
            print('Foceni uspesne dokonceno.')
        else:
            print('Foceni se nepodarilo. Zkontrolujte pripojeni kamery. Exit.')
            client.close()
            exit()

        # zde osetrit ukladani do log filu, nactene informace

        # rsync copy do lokalni slozky
        print('Kopiruju data do lokalniho stroje.')
        #vypis = os.system(lrsync)
        ftp = client.open_sftp();
        ftp.get(rpi_datadir + picname, local_datadir + picname)
        ftp.close()
        # vymyzat zdrojova data pro uvolneni prostoru
        print('Mazu zdrojova data pro uvolneni mista.')
        rrmdata = 'rm -r ' + rpi_datadir + '*'
        i,o,e = client.exec_command(rrmdata)
        exit_st2 = o.channel.recv_exit_status()
        if exit_st2 == 0:
            print('Data vymazana.')
        else:
            print('Error. Status: ' + exit_st2)
else:
    print('Nevidim kameru.. Prosím, překontrolujte připojení a restartujte skript.')
