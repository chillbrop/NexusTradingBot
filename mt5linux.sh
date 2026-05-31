#!/bin/bash
# Copyright 2000-2026, MetaQuotes Ltd.

# MetaTrader and WebView2 download urls
URL_MT5="https://download.mql5.com/cdn/web/metaquotes.software.corp/mt5/mt5setup.exe"
URL_WEBVIEW="https://msedge.sf.dl.delivery.mp.microsoft.com/filestreamingservice/files/f2910a1e-e5a6-4f17-b52d-7faf525d17f8/MicrosoftEdgeWebview2Setup.exe"

# Wine version to install: stable or devel
WINE_VERSION="staging"

# Prepare versions
. /etc/os-release

echo OS: $NAME $VERSION_ID

echo Update and install...
if [ "$NAME" = "Fedora Linux" ]; then
    echo Update system
    sudo dnf update
    sudo dnf upgrade -y

    echo Choose Wine repo
    sudo rm /etc/yum.repos.d/winehq*
    if (( $VERSION_ID >= 43 )); then
       sudo dnf config-manager addrepo --from-repofile=https://dl.winehq.org/wine-builds/fedora/43/winehq.repo
    elif (( $VERSION_ID < 43 && $VERSION_ID >= 42 )); then
       sudo dnf config-manager addrepo --from-repofile=https://dl.winehq.org/wine-builds/fedora/42/winehq.repo
    else
       sudo dnf config-manager addrepo --from-repofile=https://dl.winehq.org/wine-builds/fedora/41/winehq.repo
    fi

    echo Install Wine and Wine Mono
    sudo dnf update
    sudo dnf install winehq-$WINE_VERSION -y
    sudo dnf install wine-mono -y
else
    echo Update system
    sudo apt update
    sudo apt upgrade -y

    echo Get full version
    sudo apt install bc wget curl -y
    VERSION_FULL=$(echo "$VERSION_ID * 100" | bc -l | cut -d "." -f1)

    echo Choose Wine repo
    sudo rm /etc/apt/sources.list.d/winehq*

    sudo dpkg --add-architecture i386
    sudo mkdir -pm755 /etc/apt/keyrings
    sudo wget -O - https://dl.winehq.org/wine-builds/winehq.key | sudo gpg --dearmor -o /etc/apt/keyrings/winehq-archive.key -

    if [ "$NAME" = "Ubuntu" ]; then
        echo Ubuntu found: $NAME $VERSION_ID
        # Choose repository based on Ubuntu version
        if (( $VERSION_FULL >= 2510 )); then
           sudo wget -NP /etc/apt/sources.list.d/ https://dl.winehq.org/wine-builds/ubuntu/dists/questing/winehq-questing.sources
        elif (( $VERSION_FULL < 2510 )) && (( $VERSION_FULL >= 2504 )); then
           sudo wget -NP /etc/apt/sources.list.d/ https://dl.winehq.org/wine-builds/ubuntu/dists/plucky/winehq-plucky.sources
        elif (( $VERSION_FULL < 2410 )) && (( $VERSION_FULL >= 2400 )); then
           sudo wget -NP /etc/apt/sources.list.d/ https://dl.winehq.org/wine-builds/ubuntu/dists/noble/winehq-noble.sources
        elif (( $VERSION_FULL < 2400 )) && (( $VERSION_FULL >= 2300 )); then
           sudo wget -NP /etc/apt/sources.list.d/ https://dl.winehq.org/wine-builds/ubuntu/dists/lunar/winehq-lunar.sources
        elif (( $VERSION_FULL < 2300 )) && (( $VERSION_FULL >= 2210 )); then
           sudo wget -NP /etc/apt/sources.list.d/ https://dl.winehq.org/wine-builds/ubuntu/dists/kinetic/winehq-kinetic.sources
        elif (( $VERSION_FULL < 2210 )) && (( $VERSION_FULL >= 2100 )); then
           sudo wget -NP /etc/apt/sources.list.d/ https://dl.winehq.org/wine-builds/ubuntu/dists/jammy/winehq-jammy.sources
        elif (( $VERSION_FULL < 2100 )) && (($VERSION_FULL >= 2000 )); then
           sudo wget -NP /etc/apt/sources.list.d/ https://dl.winehq.org/wine-builds/ubuntu/dists/focal/winehq-focal.sources
        else
           sudo wget -NP /etc/apt/sources.list.d/ https://dl.winehq.org/wine-builds/ubuntu/dists/bionic/winehq-bionic.sources
        fi
    elif [ "$NAME" = "Linux Mint" ]; then
        echo Linux Mint found: $NAME $VERSION_ID
        # Choose repository based on Linux Mint version
        if (( $VERSION_FULL >= 2200 )); then
           sudo wget -NP /etc/apt/sources.list.d/ https://dl.winehq.org/wine-builds/ubuntu/dists/noble/winehq-noble.sources
        else
           sudo wget -NP /etc/apt/sources.list.d/ https://dl.winehq.org/wine-builds/ubuntu/dists/focal/winehq-focal.sources
        fi
    elif [ "$NAME" = "Debian GNU/Linux" ]; then
        echo Debian Linux found: $NAME $VERSION_ID
        # Choose repository based on Debian version
        if (( $VERSION_FULL >= 13 )); then
           sudo wget -NP /etc/apt/sources.list.d/ https://dl.winehq.org/wine-builds/debian/dists/trixie/winehq-trixie.sources
        else
           sudo wget -NP /etc/apt/sources.list.d/ https://dl.winehq.org/wine-builds/debian/dists/bookworm/winehq-bookworm.sources
        fi
    else
        echo $NAME $VERSION_ID does not supported
        exit
    fi

    echo Install Wine and Wine Mono
    sudo apt update
    sudo apt install --install-recommends winehq-$WINE_VERSION -y
fi

echo Download MetaTrader and WebView2 Runtime
curl $URL_MT5 --output mt5setup.exe
curl $URL_WEBVIEW --output webview2.exe

echo Set environment to Windows 11
WINEPREFIX=~/.mt5 winecfg -v=win11

echo Install WebView2 Runtime
WINEPREFIX=~/.mt5 wine webview2.exe /silent /install

echo Install MetaTrader 5
WINEPREFIX=~/.mt5 wine mt5setup.exe

echo Please reboot OS
