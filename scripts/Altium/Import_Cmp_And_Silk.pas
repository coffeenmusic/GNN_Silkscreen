{..............................................................................}
Procedure Run;
Const
    D = ',';
Var
    Board          : IPCB_Board;
    Dataset, Row        : TStringList;
    Cmp            : IPCB_Component;
    xorigin, yorigin, row_idx : Integer;
    ImportFile, Layer, RefDes: String;
    Silk: IPCB_Text;
    Rot: Float;
    x, y: Double;
    fileDialog: TFileOpenDialog;
Begin
    // Retrieve the current board
    Board := PCBServer.GetCurrentPCBBoard;
    If Board = Nil Then Exit;

    // Open File Dialog GUI
    fileDialog := TFileOpenDialog.Create(nil);
    fileDialog.Title := 'Please select csv prediction dataset file.';
    fileDialog.DefaultExtension := 'csv';
    if fileDialog.Execute then
       ImportFile := fileDialog.FileName
    else exit;

    Dataset := TStringList.Create;
    Dataset.LoadFromFile(ImportFile);

    xorigin := Board.XOrigin;
    yorigin := Board.YOrigin;

    PCBServer.PreProcess;

    Row := TStringList.Create;
    For row_idx := 0 To Dataset.Count-1 Do
    Begin
        Row.DelimitedText := Dataset.Get(row_idx);
        Row.StrictDelimiter := True;


        if Row.Count < 4 then
        begin
            ShowMessage('Expected 4 columns (Designator, x, y, & Rotation), got '+IntToStr(Row.Count)+'. Exiting..');
            Exit;
        end;

        RefDes := Row.Get(0);
        x := MilsToCoord(StrToFloat(Row.Get(1))) + xorigin;
        y := MilsToCoord(StrToFloat(Row.Get(2))) + yorigin;
        Rot := StrToFloat(Row.Get(3));

        Cmp := Board.GetPcbComponentByRefDes(RefDes);
        Silk := Cmp.Name;

        PCBServer.SendMessageToRobots(Silk.I_ObjectAddress, c_Broadcast, PCBM_BeginModify , c_NoEventData);

        Silk.Rotation := Rot;
        Silk.MoveToXY(x, y);

        PCBServer.SendMessageToRobots(Silk.I_ObjectAddress, c_Broadcast, PCBM_EndModify , c_NoEventData);

    End;

    PCBServer.PostProcess;

    Dataset.Free;
End;
{..............................................................................}

{..............................................................................}
