<?php

namespace App\Http\Controllers;

use App\Jobs\FileUploaded;
use App\Models\UserDocument;
use Illuminate\Http\Request;
use Illuminate\Support\Facades\Storage;

class UserDocumentController extends Controller
{
    public function store(Request $request)
    {
        $request->validate([
            'name' => 'required'
        ]);

        $path = Storage::putFile('', $request->file('textFile'), $request['name']);

        $url = Storage::url($path);

        $data = [
            'userId' => $request['userId'],
            'name' => $request['name'],
            'data' => $url
        ];

        UserDocument::create($data);

        return redirect('/users/'.$request['userId']);
    }

    public function destroy($userId, $documentId)
    {
        $userDocument = UserDocument::where('_id', $documentId)->first();

        Storage::delete($userDocument['data']);

        $userDocument -> delete();

        return redirect('/users/'.$userId);
    }
}
