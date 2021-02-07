<?php

namespace App\Http\Controllers;

use App\Models\MongoUser;
use App\Models\UserDocument;

class UserController extends Controller
{
    public function index()
    {
        $users = MongoUser::all();
        return view('user.index', compact('users'));
    }

    public function show($id)
    {
        $user = MongoUser::where('_id', $id)->first();
        #$user->load('documents');
        $documents = UserDocument::where('userId', $id)->get();
        #dd($documents);
        return view('user.show', compact('user', 'documents'));
    }
}
