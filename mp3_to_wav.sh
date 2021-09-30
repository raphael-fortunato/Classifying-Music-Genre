#!/bin/sh

count=$(ls *.mp3 | wc -l)

if [ $count  != 0 ]
then 
   for i in *.mp3; 
   do 
       ffmpeg -i "$i" "${i%.*}.wav"; 
   done 
    rm *.mp3
fi
