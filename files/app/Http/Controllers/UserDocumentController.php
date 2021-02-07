<?php

namespace App\Http\Controllers;

use App\Models\MongoUser;
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

        UserDocument::create([
            'userId' => $request['userId'],
            'name' => $request['name'],
            'data' => $json
        ]);

        return redirect('/users/'.$request['userId']);
    }

    public function destroy(Request $request)
    {

    }
}
