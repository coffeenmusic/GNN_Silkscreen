print "Loading export_dataset." 

;register this command, so we can start it from Allegro command line
axlCmdRegister( "export_dataset" `export_dataset)

;-------------------------------------
;Definition of export_dataset command
;-------------------------------------
procedure( export_dataset()
	
	axlUIWPrint(nil "Starting export_dataset script.")
	
	FILENAME = "dataset.csv"
	
	; Get Valid Silkscreen Layers
	artList = axlGetParam("artwork")->groupMembers
	artMatch = '("silk" "slk" "sleg" "pleg" "sss" "pss") ; Match these substrings to find valid artwork groups
	valid_art = '() ; valid artwork group names as strings in a list
	valid_layers = '()
	bottom_art = '()
	bs = '() ; List of all bottom silkscreen layers
	foreach(artStr artList 
		foreach(mStr artMatch
			; If substring in artwork group name and name not already added to list
			if(nindex(lowerCase(artStr) lowerCase(mStr)) != nil && member(artStr valid_art) == nil then 
				valid_art = cons(artStr valid_art)	
				artLayers = axlGetParam(strcat("artwork:" artStr))->groupMembers
				valid_layers = append(valid_layers artLayers)

				; Check if top or bottom layer. Search for bottom assume top otherwise
				foreach(layer cons(artStr artLayers)
					if(nindex(lowerCase(layer) "bottom") != nil && member(artStr bottom_art) == nil then
						valid_art = cons(artStr bottom_art)	
						bs = append(bs artLayers)
					)
				)
			)
		)
	)
	
	; 1. Change Filter Selection Set to only text, then select all
	axlClearSelSet()
	axlSetFindFilter(?enabled list("all"), ?onButtons list("noall" "text")) ; All enabled, only text checked
	axlAddSelectAll()

	; 2. Get selected object list
	textList = axlGetSelSet() ; Gets the list of object dbids in the select set.
	
	outFile = outfile(strcat("./" FILENAME) "w")
	
	fprintf(outFile "Tool,Type,Designator,x,y,L,R,T,B,Rotation,Layer,Info\n")

	; 3. Iterate text
	foreach( text textList

		if(text->parent != nil && text->parent->refdes != nil && member(text->layer valid_layers) != nil then 
			
			; String Formatting Preprocessing
			textType = if(text->text == text->parent->refdes then "slk-des" else "slk-txt")
			xStr = sprintf(nil "%f" nth(0 text->xy))
			yStr = sprintf(nil "%f" nth(1 text->xy))
			LStr = sprintf(nil "%f" nth(0 nth(0 text->bBox)))
			RStr = sprintf(nil "%f" nth(0 nth(1 text->bBox)))
			TStr = sprintf(nil "%f" nth(1 nth(1 text->bBox)))
			BStr = sprintf(nil "%f" nth(1 nth(0 text->bBox)))
			RotStr = sprintf(nil "%f" text->rotation)
			LayerStr = if(member(text->layer bs) != nil then "Bottom Layer" else "Top Layer")
			OrigLayerStr = strcat("OriginalLayer:" text->layer)
			JustifyStr = strcat("Justify:" text->justify)
			MirrorStr = strcat("IsMirror:" "FALSE")
			if(text->isMirrored != nil then
				MirrorStr = strcat("IsMirror:" "TRUE")
			)
			InfoStr = strcat(OrigLayerStr ";" JustifyStr ";" MirrorStr)
			
			; Write to File
			fprintf(outFile strcat("cadence," textType ","))
			fprintf(outFile strcat(text->parent->refdes ","))
			fprintf(outFile strcat(xStr ","))
			fprintf(outFile strcat(yStr ","))
			fprintf(outFile strcat(LStr ","))
			fprintf(outFile strcat(RStr ","))
			fprintf(outFile strcat(TStr ","))
			fprintf(outFile strcat(BStr ","))
			fprintf(outFile strcat(RotStr ","))
			fprintf(outFile strcat(LayerStr ","))
			fprintf(outFile strcat(InfoStr "\n"))
		)
	)
	
	; 4. Change Filter Selection Set to only pins, then select all
	axlClearSelSet()
	axlSetFindFilter(?enabled list("all"), ?onButtons list("noall" "pins")) ; All enabled, only text checked
	axlAddSelectAll()

	; 5. Get selected object list
	pinList = axlGetSelSet() ; Gets the list of object dbids in the select set.
	
	; 6. Iterate pins
	foreach( pin pinList

		if(pin->component != nil then 
		
			; String Formatting Preprocessing
			refdesStr = ""
			if( pin->component->name != nil then
				refdesStr = pin->component->name
			)
			PinName = strcat("PadstackName:" pin->name)
			PinNumber = strcat("PinNumber:" pin->number)
			
			netname = "None"
			if(pin->net->name != nil then 
				netname = pin->net->name
			)
			NetNameStr = strcat("NetName:" netname)
			isthrough = "FALSE"
			if(pin->isThrough != nil then
				isthrough = "TRUE"
			)
			IsThroughStr = strcat("IsThrough:" isthrough)
			
			xStr = sprintf(nil "%f" nth(0 pin->component->symbol->xy))
			yStr = sprintf(nil "%f" nth(1 pin->component->symbol->xy))
			PinXStr = strcat("PinX:" sprintf(nil "%f" nth(0 pin->xy)))
			PinYStr = strcat("PinY:" sprintf(nil "%f" nth(1 pin->xy)))
			LStr = sprintf(nil "%f" nth(0 nth(0 pin->component->bBox)))
			RStr = sprintf(nil "%f" nth(0 nth(1 pin->component->bBox)))
			TStr = sprintf(nil "%f" nth(1 nth(1 pin->component->bBox)))
			BStr = sprintf(nil "%f" nth(1 nth(0 pin->component->bBox)))
			RotStr = sprintf(nil "%f" pin->component->symbol->rotation)
			PinRotStr = strcat("PinRotation:" sprintf(nil "%f" pin->rotation))
			PinRelRotStr = strcat("PinRelativeRotation:" sprintf(nil "%f" pin->relRotation))
			layer = ""
			if(pin->component->symbol->layer != nil then
				layer = pin->component->symbol->layer
			)
			LayerStr = if(nindex(lowerCase(layer) "bottom" ) != nil then "Bottom Layer" else "Top Layer")
			OrigLayerStr = strcat("OriginalLayer:" layer)
			InfoStr = strcat(OrigLayerStr ";" PinName ";" NetNameStr ";" PinNumber ";" PinXStr ";" PinYStr ";" PinRotStr ";" PinRelRotStr ";" IsThroughStr)
			
			; Write to File
			fprintf(outFile strcat("cadence," "pin,"))
			fprintf(outFile strcat(refdesStr ","))
			fprintf(outFile strcat(xStr ","))
			fprintf(outFile strcat(yStr ","))
			fprintf(outFile strcat(LStr ","))
			fprintf(outFile strcat(RStr ","))
			fprintf(outFile strcat(TStr ","))
			fprintf(outFile strcat(BStr ","))
			fprintf(outFile strcat(RotStr ","))
			fprintf(outFile strcat(LayerStr ","))
			fprintf(outFile strcat(InfoStr "\n"))
		)
	)
	
	; 7. Change Filter Selection Set to only other segs, then select all
	axlClearSelSet()
	axlSetFindFilter(?enabled list("all"), ?onButtons list("noall" "linesegs")) ; All enabled, only other segments checked
	axlAddSelectAll()

	; 8. Get selected object list
	trackList = axlGetSelSet() ; Gets the list of object dbids in the select set.

	; 9. Iterate line segments
	foreach( trk trackList
	
		; Filter out cline segs if any
		if(trk->isEtch == nil && trk->parent->objType == "path" && member(trk->layer valid_layers) != nil then

			; String Formatting Preprocessing
			trkType = if(trk->objType == "arc" then trkType = "slk-arc" else "slk-trk")
			sym = trk->parent->parent
			refdes = if(sym->objType == "symbol" && sym->component->name != nil then refdes = sym->component->name else "")
			x1 = nth(0 nth(0 trk->startEnd))
			x2 = nth(0 nth(1 trk->startEnd))
			y1 = nth(1 nth(1 trk->startEnd))
			y2 = nth(1 nth(0 trk->startEnd))
			LStr = sprintf(nil "%f" x1)
			RStr = sprintf(nil "%f" x2)
			TStr = sprintf(nil "%f" y1)
			BStr = sprintf(nil "%f" y2)
			LayerStr = if(member(trk->layer bs) != nil then "Bottom Layer" else "Top Layer")
			OrigLayerStr = strcat("OriginalLayer:" trk->layer)
			WidthStr = strcat("Width:" sprintf(nil "%f" trk->width))
			InCompStr = strcat("InComponent:" if(refdes == "" then "0" else "-1"))
			if(trkType == "slk-trk" 
				then
					x = x1 + (x1+x2)/2
					y = y1 + (y1+y2)/2
					LineTypeStr = strcat("LineType:" trk->lineType)
					InfoStr = strcat(OrigLayerStr ";" WidthStr ";" InCompStr ";" LineTypeStr)
				else
					x = nth(0 trk->xy)
					y = nth(1 trk->xy)
					isCircle = if(trk->isCircle == nil then "UNCLOSED_ARC" else "CIRCLE")
					CircleStr = strcat("isCircle:" isCircle)
					isClock = if(trk->isClockwise == nil then "FALSE" else "TRUE")
					ClockStr = strcat("isClockwise:" isClock)
					RadiusStr = strcat("Radius:" sprintf(nil "%f" trk->radius))
					InfoStr = strcat(OrigLayerStr ";" WidthStr ";" InCompStr ";" RadiusStr ";" CircleStr ";" ClockStr)
			)
			xStr = sprintf(nil "%f" x)
			yStr = sprintf(nil "%f" y)
			
			
			; Write to File
			fprintf(outFile strcat("cadence," trkType ","))
			fprintf(outFile strcat(refdes ","))
			fprintf(outFile strcat(xStr ","))
			fprintf(outFile strcat(yStr ","))
			fprintf(outFile strcat(LStr ","))
			fprintf(outFile strcat(RStr "0,"))
			fprintf(outFile strcat(TStr ","))
			fprintf(outFile strcat(BStr ","))
			fprintf(outFile strcat(RotStr ","))
			fprintf(outFile strcat(LayerStr ","))
			fprintf(outFile strcat(InfoStr "\n"))
		)
	)
	
	; 10. Close the file and wrap up
	close(outFile)
	
	; 11. This next line launches Excel and displays the report
	axlHttp(FILENAME)

)