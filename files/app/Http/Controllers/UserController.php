<?php

namespace App\Http\Controllers;

use App\Models\MongoUser;

class UserController extends Controller
{
    public function index()
    {
        $users = MongoUser::all();
        return view('user.index', compact('users'));
    }
}
