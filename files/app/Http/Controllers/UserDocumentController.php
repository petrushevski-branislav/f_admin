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
            'textFile' => 'mimes:txt' // Only allow .txt file types
        ]);

        $uploadedFile = $request->file('textFile');
        $json = $uploadedFile->get();

        UserDocument::create([
            'userId' => $request['userId'],
            'data' => $json
        ]);

        $users = MongoUser::all();

        return view('user.index', compact('users'));
    }

    public function destroy(Request $request)
    {

    }
}
