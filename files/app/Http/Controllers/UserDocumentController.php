<?php

namespace App\Http\Controllers;

use App\Jobs\FileUploaded;
use App\Models\UserDocument;
use Illuminate\Http\Request;

class UserDocumentController extends Controller
{
    public function store(Request $request)
    {
        $request->validate([
            'textFile' => 'mimes:txt', // Only allow .txt file types
            'name' => 'required'
        ]);

        $uploadedFile = $request->file('textFile');
        $json = $uploadedFile->get();

        $data = [
            'userId' => $request['userId'],
            'name' => $request['name'],
            'data' => $json
        ];

        UserDocument::create($data);

        FileUploaded::dispatch($data);

        return redirect('/users/'.$request['userId']);
    }

    public function destroy($userId, $documentId)
    {
        $userDocument = UserDocument::where('_id', $documentId)->first();

        $userDocument -> delete();

        return redirect('/users/'.$userId);
    }
}
