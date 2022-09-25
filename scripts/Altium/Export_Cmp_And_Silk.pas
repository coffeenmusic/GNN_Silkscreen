const
    TOOL = 'altium';

function GetPins(Board: IPCB_Board, Dataset: TStringList, Cmp: IPCB_Component, CmpData: String): String;
const
    D = ',';
var
    GrpIter : IPCB_GroupIterator;
    Pad : IPCB_Pad;
    Nets : TStringList;
    x, y, NetName, Info: String;
    xorigin, yorigin: Integer;
begin
    GrpIter := Cmp.GroupIterator_Create;
    GrpIter.SetState_FilterAll;
    GrpIter.AddFilter_ObjectSet(MkSet(ePadObject));

    Nets := TStringList.Create;
    Nets.Delimiter := ';';
    Nets.Sorted := True;
    Nets.Duplicates := dupIgnore;

    xorigin := Board.XOrigin;
    yorigin := Board.YOrigin;

    Pad := GrpIter.FirstPCBObject;
    While Pad <> Nil Do
    Begin
        x := FloatToStr(CoordToMils(Pad.x - xorigin));
        y := FloatToStr(CoordToMils(Pad.y - yorigin));

        If Pad.Net <> Nil Then
        Begin
            NetName := Pad.Net.Name;
        End
        Else
        Begin
            NetName := 'None';
        End;

        Info := 'PinName:'+Pad.Name;
        Info := Info + ';NetName:'+NetName;
        Info := Info + ';PinX:'+x;
        Info := Info + ';PinY:'+y;

        Dataset.Add(CmpData+D+Info);

        Pad := GrpIter.NextPCBObject;
    End;
    Cmp.GroupIterator_Destroy(GrpIter);
end;

function GetSilkTracks(Board: IPCB_Board, Dataset: TStringList): TStringList;
const
    D = ',';
var
    Iterator       : IPCB_BoardIterator;
    xo, yo, x, y, x1, x2, y1, y2: Integer;
    Obj: IPCB_ObjectClass;
    TypeName, designator, layer, Info: String;
begin
    xo := Board.XOrigin;
    yo := Board.YOrigin;

    Iterator := Board.BoardIterator_Create;
    Iterator.AddFilter_ObjectSet(MkSet(eTrackObject, eArcObject));
    Iterator.AddFilter_IPCB_LayerSet(MkSet(eTopOverlay, eBottomOverlay));
    Iterator.AddFilter_Method(eProcessAll);

    Obj := Iterator.FirstPCBObject;
    While (Obj <> Nil) Do
    Begin
        layer := Layer2String(Obj.Layer);

        if (layer <> 'Top Overlay') and (layer <> 'Bottom Overlay') then
        begin
            Obj := Iterator.NextPCBObject;
            continue;
        end;

        designator := '';
        if Obj.InComponent then designator := Obj.Component.Name.Text;

        layer := 'Top Layer';
        if Layer2String(Obj.Layer) = 'Bottom Overlay' then layer := 'Bottom Layer';

        Info := '';
        if Obj.ObjectId = eTrackObject then // Tracks
        begin
            TypeName := 'slk-trk';

            x1 := CoordToMils(Obj.x1 - xo);
            x2 := CoordToMils(Obj.x2 - xo);
            y1 := CoordToMils(Obj.y1 - yo);
            y2 := CoordToMils(Obj.y2 - yo);
            x := x1 + (x2 - x1)/2;
            y := y1 + (y2 - y1)/2;

            Info := 'Width:'+FloatToStr(CoordToMils(Obj.Width));
            Info := Info + ';Length:'+FloatToStr(CoordToMils(Obj.GetState_Length()));
        end
        else if Obj.ObjectId = eArcObject then // Arcs
        begin
            TypeName := 'slk-arc';

            x := CoordToMils(Obj.XCenter - xo);
            y := CoordToMils(Obj.YCenter - yo);
            x1 := CoordToMils(Obj.StartX - xo);
            y1 := CoordToMils(Obj.StartY - yo);
            x2 := CoordToMils(Obj.EndX - xo);
            y2 := CoordToMils(Obj.EndY - yo);

            Info := 'Width:'+FloatToStr(CoordToMils(Obj.LineWidth));
            Info := Info + ';Radius:'+FloatToStr(CoordToMils(Obj.Radius));
            Info := Info + ';StartAngle:'+FloatToStr(Obj.StartAngle);
            Info := Info + ';EndAngle:'+FloatToStr(Obj.EndAngle);
        end;
        Info := Info + ';InComponent:'+BoolToStr(Obj.InComponent);

        Dataset.Add(TOOL+D+TypeName+D+designator+D+FloatToStr(x)+D+FloatToStr(y)+D+
        FloatToStr(x1)+D+FloatToStr(x2)+D+FloatToStr(y1)+D+FloatToStr(y2)+D+
        '0'+D+layer+D+Info);
        Obj := Iterator.NextPCBObject;
    End;
    Board.BoardIterator_Destroy(Iterator);

    result := Dataset;
end;

{..............................................................................}
Procedure Run;
Const
    DEFAULT_FILE = 'dataset.csv';
    D = ',';
    SilkOnly = False;
Var
    Board          : IPCB_Board;
    Iterator       : IPCB_BoardIterator;
    ReportDocument : IServerDocument;
    FileName       : TPCBString;
    Document       : IServerDocument;
    Dataset        : TStringList;
    Cmp            : IPCB_Component;
    xorigin, yorigin : Integer;
    Layer, x,y,L,R,T,B, RefDes, Rot, Nets, CmpData, SavePath: String;
    Rect: TCoordRect;
    Silk: IPCB_Text;
    saveDialog : TSaveDialog;
Begin
    // Retrieve the current board
    Board := PCBServer.GetCurrentPCBBoard;
    If Board = Nil Then Exit;

    Iterator := Board.BoardIterator_Create;
    Iterator.AddFilter_ObjectSet(MkSet(eComponentObject));
    Iterator.AddFilter_IPCB_LayerSet(LayerSet.AllLayers);
    Iterator.AddFilter_Method(eProcessAll);

    Dataset := TStringList.Create;
    Dataset.Add('Tool,Type,Designator,x,y,L,R,T,B,Rotation,Layer,Info');
    // Note: Info is an additional delimited list of pin, track, cmp, etc. specific information

    // Open Save File Dialog GUI
    saveDialog := TSaveDialog.Create(nil);
    saveDialog.Title := 'Please select a location and filename for the saved dataset.';
    saveDialog.Filter := 'CSV file|*.csv';
    saveDialog.DefaultExt := 'csv';
    saveDialog.FilterIndex := 0;
    saveDialog.FileName := DEFAULT_FILE;
    if saveDialog.Execute then
       SavePath := saveDialog.FileName
    else exit;

    xorigin := Board.XOrigin;
    yorigin := Board.YOrigin;

    Cmp := Iterator.FirstPCBObject;
    While (Cmp <> Nil) Do
    Begin
        Layer := Layer2String(Cmp.Layer);

        if SilkOnly = False then
        begin
            // Component
            RefDes := Cmp.Name.Text;
            x := FloatToStr(CoordToMils(Cmp.x - xorigin));
            y := FloatToStr(CoordToMils(Cmp.y - yorigin));
            Rot := FloatToStr(Cmp.Rotation);
            Rect := Cmp.BoundingRectangleNoNameComment;
            L := FloatToStr(CoordToMils(Rect.Left - xorigin));
            R := FloatToStr(CoordToMils(Rect.Right - xorigin));
            T := FloatToStr(CoordToMils(Rect.Top - yorigin));
            B := FloatToStr(CoordToMils(Rect.Bottom - yorigin));

            CmpData := TOOL+D+'pin'+D+RefDes+D+x+D+y+D+L+D+R+D+T+D+B+D+Rot+D+Layer;

            // Get Each Pin & Net Name
            GetPins(Board, Dataset, Cmp, CmpData);
        end;

        // Silkscreen
        Silk := Cmp.Name;
        x := FloatToStr(CoordToMils(Silk.XLocation - xorigin));
        y := FloatToStr(CoordToMils(Silk.YLocation - yorigin));
        Rect := Silk.BoundingRectangle;
        L := FloatToStr(CoordToMils(Rect.Left - xorigin));
        R := FloatToStr(CoordToMils(Rect.Right - xorigin));
        T := FloatToStr(CoordToMils(Rect.Top - yorigin));
        B := FloatToStr(CoordToMils(Rect.Bottom - yorigin));
        Rot := FloatToStr(Silk.Rotation);

        // Don't add hidden silkscreen
        If Cmp.NameOn Then
        Begin
             Dataset.Add(TOOL+D+'slk-des'+D+Silk.Text+D+x+D+y+D+L+D+R+D+T+D+B+D+Rot+D+Layer+D+'NA');
        End;

        Cmp := Iterator.NextPCBObject;
    End;
    Board.BoardIterator_Destroy(Iterator);

    // Append all silk track & arcs to the dataset
    Dataset := GetSilkTracks(Board, Dataset);

    Dataset.SaveToFile(SavePath);
    Dataset.Free;
End;
{..............................................................................}

{..............................................................................}
